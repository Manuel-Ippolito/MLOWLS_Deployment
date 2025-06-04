import os
import tempfile
import logging
import types
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
# from ml_owls.auth import get_api_key
import time

import requests

# Change the import to use the pipeline_instance module
from ml_owls.model.inference.pipeline_singleton import initialize_pipeline, labelstudio_url
from ml_owls.model.inference.predict import predict_single_file
from ml_owls.configs.config import load_config
from ml_owls.labelstudio_integration.add_prediction import send_to_labelstudio
from utils.id_to_common_name import primary_id_to_common_name


router = APIRouter()
logger = logging.getLogger(__name__)

pipeline = initialize_pipeline()


@router.get("/")
def read_root():
    return {"message": "FastAPI is up!"}


@router.get("/liveness")
def liveness_check():
    """
    Simple liveness check endpoint to verify the application is alive.
    This endpoint can be used by load balancers or health check systems to ensure
    the application is running and responsive.
    """
    return {
        "status": "alive",
        "timestamp": time.time()
    }


@router.get("/readiness")
def readiness_check():
    """
    Readiness check endpoint to verify the application can serve requests.
    Checks if the model is loaded and dependencies are available.
    """
    status = {"status": "ready", "checks": {}}
    
    # Check if the model inference pipeline is initialized
    if pipeline is None:
        status["status"] = "not_ready"
        status["checks"]["model"] = "not_loaded"
    else:
        status["checks"]["model"] = "loaded"
    
    # Check Label Studio connection if it's being used
    if labelstudio_url:
        try:
            # Simple HEAD request to check if Label Studio is reachable
            response = requests.head(
                labelstudio_url, 
                timeout=2
            )
            if response.status_code < 400:
                status["checks"]["labelstudio"] = "connected"
            else:
                status["status"] = "not_ready"
                status["checks"]["labelstudio"] = f"error_status_{response.status_code}"
        except Exception as e:
            status["status"] = "not_ready"
            status["checks"]["labelstudio"] = f"connection_error: {str(e)}"
    else:
        status["checks"]["labelstudio"] = "not_configured"
    
    return status


@router.post("/predict")
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

            # Convert species IDs to common names in the prediction results
            if prediction and prediction.get('predictions'):
                for pred in prediction['predictions']:
                    try:
                        species_id = pred.get('species_name', '0')
                        pred['species_name'] = primary_id_to_common_name(species_id)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Could not convert species ID {pred.get('species_name')} to common name: {e}")
                        pred['species_name'] = f"Unknown Species (ID: {pred.get('species_name', 'N/A')})"


            # Add the prediction and confidence to Label Studio if configured
            labelstudio_result = None
            if labelstudio_url:
                try:
                    # Extract the top prediction from the results
                    if prediction and prediction.get('predictions'):
                        top_prediction = prediction['predictions'][0]  # Get the highest confidence prediction
                        species_name = top_prediction.get('species_name', 'unknown')
                        confidence = top_prediction.get('confidence', 0.0)
                        labelstudio_result = send_to_labelstudio(filename=file.filename,
                                                                 prediction=species_name,
                                                                 confidence=float(confidence))
                        
                        # Check if the result indicates an error
                        if not labelstudio_result.get("success", False):
                            logger.warning(f"Label Studio integration failed: {labelstudio_result.get('error', 'Unknown error')}")
                        else:
                            logger.info(f"Successfully sent prediction to Label Studio: {species_name}")
                    else:
                        labelstudio_result = {"success": False, "error": "No predictions to send"}
                        logger.warning("No predictions available to send to Label Studio")
                except Exception as e:
                    logger.error(f"Failed to send to Label Studio: {str(e)}")
                    labelstudio_result = {"success": False, "error": str(e)}
            
            # Return prediction results with Label Studio integration info
            response = {"prediction": prediction}
            if labelstudio_result:
                response["labelstudio_result"] = labelstudio_result
                
            return response
        
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
