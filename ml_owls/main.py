# app/main.py
import logging
from fastapi import FastAPI
from ml_owls.router.endpoints import router, onnx_session, label_map, sample_rate, labelstudio_url, labelstudio_token
from ml_owls.config import load_config
from ml_owls.model.onnx_model import load_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

config = load_config()

# Initialize model and config
onnx_session = load_model(config["model"]["path"])
label_map.update({str(k): v for k, v in config["model"]["labels"].items()})
sample_rate = config["model"]["sample_rate"]
labelstudio_url = config["labelstudio"]["url"]
labelstudio_token = config["labelstudio"]["token"]

# Create FastAPI app
app = FastAPI(
    title=config["app"]["name"],
    version="1.0.0",
    description="BirdCLEF 2025 Inference API"
)

# Include routes
app.include_router(router)
