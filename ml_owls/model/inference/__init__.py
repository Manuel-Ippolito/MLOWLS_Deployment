# src/inference/__init__.py
"""
Inference pipeline for BirdCLEF models.
"""

from ml_owls.model.inference.convert import main as convert_main
from ml_owls.model.inference.model_converter import ModelConverter

__all__ = ["ModelConverter", "convert_main"]
