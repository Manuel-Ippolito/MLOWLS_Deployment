"""
Interfaces package for defining abstract contracts.
"""

from .logger import Logger
from .metric_calculator import MetricCalculator
from .model_saver import ModelSaver
from .training_strategy import TrainingStrategy

__all__ = ["Logger", "MetricCalculator", "ModelSaver", "TrainingStrategy"]
