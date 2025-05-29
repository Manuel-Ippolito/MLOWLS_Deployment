import logging
import os
from fastapi import FastAPI
import ml_owls.router as router_module
from ml_owls.router import router
from ml_owls.configs.config import load_config
from ml_owls.model.onnx_model import load_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

config = load_config()

# Initialize model and config
router_module.onnx_session = load_model(config["model"]["path"])
router_module.labelstudio_url = config["labelstudio"]["url"]
router_module.labelstudio_token = os.getenv("LABELSTUDIO_TOKEN", "")

# Create FastAPI app
app = FastAPI(
    title=config["app"]["name"],
    version="1.0.0",
    description="BirdCLEF 2025 Inference API"
)

# Include routes
app.include_router(router)