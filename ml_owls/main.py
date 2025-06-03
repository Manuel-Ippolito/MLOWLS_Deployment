import logging
import os
from fastapi import FastAPI
from ml_owls.router import router
from ml_owls.configs.config import load_config
from ml_owls.model.inference.pipeline_singleton import initialize_pipeline as initialize_inference_pipeline
from label_studio_sdk.client import LabelStudio

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

# Read the label configuration from file
def load_label_config():
    config_path = 'data/ls_taxonomy.txt'
    with open(config_path, 'r', encoding='utf-8') as f:
        return f.read()

# Connect to the Label Studio API and check the connection
labelstudio_client = LabelStudio(base_url=os.getenv("LABELSTUDIO_URL"), api_key=os.getenv("LABELSTUDIO_TOKEN"))

# Create a project with the taxonomy-based labeling configuration
try:
    # Try to get existing project first
    projects = labelstudio_client.projects.list()
    project = None
    for p in projects:
        if p.title == 'ml_owls':
            project = p
            break
    if not project:
        # Create new project if it doesn't exist
        label_config = load_label_config()
        project = labelstudio_client.projects.create(
            title='ml_owls',
            label_config=label_config
        )
        logger.info(f"Created new Label Studio project: {project.id}")
    else:
        logger.info(f"Using existing Label Studio project: {project.id}")
        
except Exception as e:
    logger.error(f"Error setting up Label Studio project: {e}")
    project = None

# Create FastAPI app
app = FastAPI(
    title=config["app"]["name"],
    version="1.0.0",
    description="BirdCLEF 2025 Inference API"
)

# Include routes
app.include_router(router)
