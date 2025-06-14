import logging
import os
from fastapi import FastAPI
from ml_owls.router import router
from ml_owls.configs.config import load_config
from ml_owls.model.inference.pipeline_singleton import initialize_pipeline as initialize_inference_pipeline
from ml_owls.labelstudio_integration.labelstudio_singleton import initialize_labelstudio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load config
config = load_config()

# Initialize the inference pipeline
initialize_inference_pipeline()

# Initialize Label Studio
initialize_labelstudio()

# Create FastAPI app
app = FastAPI(
    title=config["app"]["name"],
    version="1.0.0",
    description="BirdCLEF 2025 Inference API"
)

# Include routes
app.include_router(router)