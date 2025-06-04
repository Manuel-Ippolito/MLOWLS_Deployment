import logging
import os
from label_studio_sdk.client import LabelStudio

logger = logging.getLogger(__name__)

# Global variables for Label Studio client and project
labelstudio_client = None
project = None

def initialize_labelstudio():
    """Initialize Label Studio client and project."""
    global labelstudio_client, project
    
    try:
        # Connect to the Label Studio API
        labelstudio_client = LabelStudio(
            base_url=os.getenv("LABELSTUDIO_URL"), 
            api_key=os.getenv("LABELSTUDIO_TOKEN")
        )
        
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
        labelstudio_client = None

def load_label_config():
    """Load the label configuration from file."""
    config_path = 'data/ls_taxonomy.txt'
    with open(config_path, 'r', encoding='utf-8') as f:
        return f.read()
