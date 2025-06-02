import os
import logging
from ml_owls.model.inference.inference_factory import InferenceFactory
from ml_owls.configs.config import load_config

logger = logging.getLogger(__name__)


# Initialize global variables
pipeline = None
model_metadata_path = None
model_top_k = 5
model_confidence_threshold = 0.1
use_gpu = False
labelstudio_url = "http://labelstudio"
labelstudio_port = 8080


def initialize_pipeline():
    """Initialize the inference pipeline from config."""
    global pipeline, model_metadata_path, model_top_k, model_confidence_threshold, use_gpu
    global labelstudio_url, labelstudio_port
    
    config = load_config()
    
    # Set configuration values
    model_metadata_path = config["model"]["metadata_path"]
    model_top_k = config["model"]["top_k"]
    model_confidence_threshold = config["model"]["confidence_threshold"]
    use_gpu = config["model"]["use_gpu"]
    labelstudio_url = config["labelstudio"]["url"]
    labelstudio_port = config["labelstudio"]["port"]
    
    # Initialize pipeline
    logger.info(f"Initializing inference pipeline with model metadata: {model_metadata_path}")
    pipeline = InferenceFactory.create_from_converted_model(
        conversion_metadata_path=model_metadata_path,
        top_k=model_top_k,
        confidence_threshold=model_confidence_threshold,
        use_gpu=use_gpu
    )
    logger.info("Inference pipeline initialized successfully")
    
    return pipeline
