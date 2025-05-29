# app/config.py
import yaml
from pathlib import Path

def load_config(config_path: str = "configs/ml_owls_config.yml"):
    with open(Path(config_path), "r") as f:
        return yaml.safe_load(f)
