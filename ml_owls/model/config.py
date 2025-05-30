from typing import Any

import yaml


class Config:
    """Load and store experiment configuration from a YAML file."""

    def __init__(self, path: str = "configs/model/model_config.yaml") -> None:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # Paths configuration
        paths = cfg.get("paths", {})
        self.train_csv: str = paths["train_csv"]
        self.train_audio_dir: str = paths["train_audio_dir"]
        self.taxonomy_csv: str = paths["taxonomy_csv"]
        self.output_dir: str = paths.get("output_dir", "outputs")

        # Audio configuration
        audio_cfg = cfg.get("audio", {})
        self.sample_rate: int = audio_cfg.get("sample_rate", 32000)
        self.n_mels: int = audio_cfg.get("n_mels", 128)
        self.n_fft: int = audio_cfg.get("n_fft", 1024)
        self.hop_length: int = audio_cfg.get("hop_length", 320)
        self.fmin: int = audio_cfg.get("fmin", 20)
        self.fmax: int = audio_cfg.get("fmax", 16000)
        self.segment_length: float = float(audio_cfg.get("segment_length", 30.0))
        self.overlap: float = float(audio_cfg.get("overlap", 0.5))
        self.max_segments_per_file: int = audio_cfg.get("max_segments_per_file", 5)

        # Training configuration
        train_cfg = cfg.get("training", {})
        self.batch_size: int = train_cfg.get("batch_size", 32)
        self.epochs: int = train_cfg.get("epochs", 10)
        self.learning_rate: float = float(train_cfg.get("learning_rate", 1e-3))
        self.weight_decay: float = float(train_cfg.get("weight_decay", 1e-4))
        self.val_fraction: float = train_cfg.get("val_fraction", 0.1)
        self.mixup_alpha: float = float(train_cfg.get("mixup_alpha", 0.4))
        self.label_smoothing: float = float(train_cfg.get("label_smoothing", 0.1))

        # Model configuration
        model_cfg = cfg.get("model", {})
        self.backbone: str = model_cfg.get("backbone", "efficientnet_b0")
        self.num_classes: int = model_cfg.get("num_classes", 206)
        self.dropout: float = float(model_cfg.get("dropout", 0.3))

        # Experiment configuration
        exp_cfg = cfg.get("experiment", {})
        self.experiment_name: str = exp_cfg.get("name", "birdclef_experiment")
        self.run_name: str | None = exp_cfg.get("run_name", None)
        self.experiment_tags: dict[str, Any] = exp_cfg.get("tags", {})

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return self.__dict__

    def save(self, path: str) -> None:
        """Save config to YAML file."""
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f)
