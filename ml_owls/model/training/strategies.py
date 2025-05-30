from typing import Tuple

import torch

from interfaces.model.training_strategy import TrainingStrategy
from utils.model.utils import mixup_data


class StandardTrainingStrategy(TrainingStrategy):
    """Standard training without data augmentation."""

    @property
    def name(self) -> str:
        return "standard"

    def train_step(
        self,
        model: torch.nn.Module,
        batch: Tuple[torch.Tensor, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute standard training step.

        Args:
            model: Neural network model
            batch: Tuple of (inputs, labels)
            optimizer: Optimizer instance
            criterion: Loss function
            device: Device to run computation on

        Returns:
            Tuple of (loss, predictions, original_labels)
        """
        specs, labels = batch
        specs, labels = specs.to(device), labels.to(device)

        optimizer.zero_grad()
        predictions = model(specs)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        return loss, predictions, labels


class MixupTrainingStrategy(TrainingStrategy):
    """Training with mixup data augmentation."""

    def __init__(self, alpha: float = 0.4):
        """Initialize mixup strategy.

        Args:
            alpha: Mixup interpolation strength parameter
        """
        self.alpha = alpha
        self.fallback_strategy = StandardTrainingStrategy()

    @property
    def name(self) -> str:
        return f"mixup_alpha_{self.alpha}"

    def train_step(
        self,
        model: torch.nn.Module,
        batch: Tuple[torch.Tensor, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute mixup training step.

        Args:
            model: Neural network model
            batch: Tuple of (inputs, labels)
            optimizer: Optimizer instance
            criterion: Loss function
            device: Device to run computation on

        Returns:
            Tuple of (loss, predictions, original_labels)
        """
        specs, labels = batch
        specs, labels = specs.to(device), labels.to(device)

        # Apply mixup only during training and if alpha > 0
        if self.alpha > 0 and model.training:
            mixed_specs, targets_a, targets_b, lam = mixup_data(specs, labels, self.alpha)

            optimizer.zero_grad()
            predictions = model(mixed_specs)

            # Mixup loss calculation
            loss = lam * criterion(predictions, targets_a) + (1 - lam) * criterion(
                predictions, targets_b
            )

            loss.backward()
            optimizer.step()

            # Return original labels for metric calculation
            return loss, predictions, labels
        else:
            # Fallback to standard training
            return self.fallback_strategy.train_step(model, batch, optimizer, criterion, device)


class CutMixTrainingStrategy(TrainingStrategy):
    """Training with CutMix data augmentation for spectrograms."""

    def __init__(self, alpha: float = 1.0):
        """Initialize CutMix strategy.

        Args:
            alpha: CutMix interpolation strength parameter
        """
        self.alpha = alpha
        self.fallback_strategy = StandardTrainingStrategy()

    @property
    def name(self) -> str:
        return f"cutmix_alpha_{self.alpha}"

    def train_step(
        self,
        model: torch.nn.Module,
        batch: Tuple[torch.Tensor, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute CutMix training step.

        Args:
            model: Neural network model
            batch: Tuple of (inputs, labels)
            optimizer: Optimizer instance
            criterion: Loss function
            device: Device to run computation on

        Returns:
            Tuple of (loss, predictions, original_labels)
        """
        specs, labels = batch
        specs, labels = specs.to(device), labels.to(device)

        # Apply CutMix only during training and if alpha > 0
        if self.alpha > 0 and model.training:
            cut_specs, targets_a, targets_b, lam = self._cutmix_data(specs, labels)

            optimizer.zero_grad()
            predictions = model(cut_specs)

            # CutMix loss calculation
            loss = lam * criterion(predictions, targets_a) + (1 - lam) * criterion(
                predictions, targets_b
            )

            loss.backward()
            optimizer.step()

            return loss, predictions, labels
        else:
            return self.fallback_strategy.train_step(model, batch, optimizer, criterion, device)

    def _cutmix_data(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply CutMix augmentation to spectrogram data.

        Args:
            x: Input spectrograms of shape (batch_size, channels, freq, time)
            y: Labels

        Returns:
            Tuple of (mixed_x, y_a, y_b, lambda)
        """
        if self.alpha > 0:
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        else:
            lam = 1.0

        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)

        # Get random cut coordinates
        _, _, H, W = x.shape
        cut_rat = torch.sqrt(1.0 - lam)
        cut_w = torch.randint(0, int(W * cut_rat) + 1, (1,)).item()
        cut_h = torch.randint(0, int(H * cut_rat) + 1, (1,)).item()

        cx = torch.randint(0, W, (1,)).item()
        cy = torch.randint(0, H, (1,)).item()

        bbx1 = max(0, cx - cut_w // 2)
        bby1 = max(0, cy - cut_h // 2)
        bbx2 = min(W, cx + cut_w // 2)
        bby2 = min(H, cy + cut_h // 2)

        # Apply CutMix
        x_cut = x.clone()
        x_cut[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

        # Adjust lambda based on actual cut area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

        return x_cut, y, y[index], lam
