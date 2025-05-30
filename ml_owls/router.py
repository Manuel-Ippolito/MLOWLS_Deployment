import os
import tempfile
import logging
import types
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from ml_owls.auth import get_api_key
import requests

# Change the import to use the pipeline_instance module
from ml_owls.model.inference.pipeline_singleton import initialize_pipeline, labelstudio_token, labelstudio_url
from ml_owls.model.inference.predict import predict_single_file
from ml_owls.configs.config import load_config

router = APIRouter()
logger = logging.getLogger(__name__)

pipeline = initialize_pipeline()

@router.get("/health")
def health_check():
    return {"status": "OK"}


@router.get("/readiness")
def readiness_check():
    return {"status": "ready"}


@router.post("/predict")  #, dependencies=[Depends(get_api_key)])
async def predict_endpoint(file: UploadFile = File(...)):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Create a temporary file to save the uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            # Read uploaded file content
            content = await file.read()
            # Write content to temporary file
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            # Create a simple namespace to pass as args
            config = load_config()
            args = types.SimpleNamespace(
                aggregate="max",  # Aggregation method
                config=config["model"]["config_path"],  # Path to model config
                quiet=False,  # Show output
                detailed=False  # Don't show detailed output
            )
            
            # Process the audio file using the inference pipeline
            prediction = predict_single_file(
                pipeline=pipeline,
                audio_path=temp_path,
                args=args
            )
            
            # Return prediction results
            return {"prediction": prediction}
        
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/label", dependencies=[Depends(get_api_key)])
async def label_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        files = {"file": (file.filename, contents)}
        headers = {"Authorization": f"Token {labelstudio_token}"}
        response = requests.post(labelstudio_url, files=files, headers=headers)
        return {"labelstudio_response": response.json()}
    except Exception as e:
        logger.error(f"LabelStudio error: {str(e)}")
        raise HTTPException(status_code=500, detail="LabelStudio interaction failed")
