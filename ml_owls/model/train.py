import argparse
import os

import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from ml_owls.model.config import Config
from ml_owls.model.dataset import BirdClefDataset, collate_fn
from ml_owls.model.mlflow_logger import MLflowLogger
from ml_owls.model.trainer_factory import TrainerFactory
from utils.model.utils import get_mel_log_transform


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy for robust training with weak labels."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Apply label smoothing."""
        confidence = 1.0 - self.smoothing
        log_probs = torch.nn.functional.log_softmax(pred, dim=1)
        nll_loss = torch.nn.functional.nll_loss(log_probs, target, reduction="mean")
        smooth_loss = -log_probs.mean(dim=1).mean()
        return confidence * nll_loss + self.smoothing * smooth_loss


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train BirdCLEF model")
    parser.add_argument("--config", type=str, default="configs/model/model_config.yaml", help="Path to config file")
    return parser.parse_args()


def main() -> None:
    """Main training function using new SOLID architecture."""
    args = parse_args()
    cfg = Config(args.config)

    # 1) Load metadata and map labels via taxonomy
    df = pd.read_csv(cfg.train_csv)
    tax_df = pd.read_csv(cfg.taxonomy_csv)
    code2idx = {code: idx for idx, code in enumerate(tax_df["primary_label"])}
    df["label_idx"] = df["primary_label"].map(code2idx)

    print(f"Loaded {len(df)} samples across {len(code2idx)} species")

    # 2) Split train/val
    if cfg.val_fraction > 0:
        train_df, val_df = train_test_split(
            df,
            test_size=cfg.val_fraction,
            stratify=df["label_idx"],
            random_state=42,
        )
        print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    else:
        train_df, val_df = df, None
        print(f"Train: {len(train_df)}, No validation split")

    # 3) Prepare transforms
    mel_transform = get_mel_log_transform(
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        power=2.0,
    )

    # 4) Build DataLoaders
    train_ds = BirdClefDataset(
        train_df, cfg.train_audio_dir, config=cfg, transform=mel_transform, is_train=True
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    dataloaders = {"train": train_loader}

    if val_df is not None:
        val_ds = BirdClefDataset(
            val_df, cfg.train_audio_dir, config=cfg, transform=mel_transform, is_train=False
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        dataloaders["val"] = val_loader

    # 5) Instantiate model, loss, optimizer
    model = timm.create_model(
        cfg.backbone,
        pretrained=True,
        num_classes=cfg.num_classes,
        in_chans=1,
        drop_rate=cfg.dropout,
    )

    criterion = LabelSmoothingCrossEntropy(smoothing=cfg.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    # 6) Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 7) Create output directory
    os.makedirs(cfg.output_dir, exist_ok=True)

    # 8) Start training
    with MLflowLogger(cfg.experiment_name, cfg.run_name) as logger:
        # Log configuration
        logger.log_config(cfg)

        # Log dataset stats
        train_stats = train_ds.get_stats()
        logger.log_dataset_stats(train_stats)

        if val_df is not None:
            val_stats = val_ds.get_stats()
            logger.log_dataset_stats({f"val_{k}": v for k, v in val_stats.items()})

        # Create trainer using factory
        trainer = TrainerFactory.create_trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            dataloaders=dataloaders,
            device=device,
            config=cfg,
            logger=logger,
        )

        # Start training
        trainer.fit(cfg.epochs)

    print("Training completed!")


if __name__ == "__main__":
    main()
