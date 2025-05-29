# app/config.py
import yaml
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str = "configs/ml_owls_config.yml"):
    with open(Path(config_path), "r") as f:
        return yaml.safe_load(f)
