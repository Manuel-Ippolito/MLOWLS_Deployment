import logging
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
router_module.label_map = {str(k): v for k, v in config["model"]["labels"].items()}
router_module.sample_rate = config["model"]["sample_rate"]
router_module.labelstudio_url = config["labelstudio"]["url"]
router_module.labelstudio_token = config["labelstudio"]["token"]

# Create FastAPI app
app = FastAPI(
    title=config["app"]["name"],
    version="1.0.0",
    description="BirdCLEF 2025 Inference API"
)

# Include routes
app.include_router(router)