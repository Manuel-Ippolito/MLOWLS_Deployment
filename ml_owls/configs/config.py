# app/config.py
import yaml
from pathlib import Path

def load_config(config_path: str = "config.yaml"):
    with open(Path(config_path), "r") as f:
        return yaml.safe_load(f)
