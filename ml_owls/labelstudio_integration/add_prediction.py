import logging
from label_studio_sdk.label_interface.objects import PredictionValue

from utils.labelstudio_integration.find_audio import find_ogg_file_path

logger = logging.getLogger(__name__)

def send_to_labelstudio(filename: str, prediction: str, confidence: float):
    """
    Send prediction results to Label Studio.
    
    Args:
        filename (str): Name of the audio file.
        prediction (str): Predicted species name (must match one of the labels in taxonomy).
        confidence (float): Confidence score of the prediction (0.0 to 1.0).
  
    Returns:
        dict: Response containing task_id and prediction_id, or error information.
    """
    from ml_owls.labelstudio_integration.labelstudio_singleton import labelstudio_client
    try:
        project = labelstudio_client.projects.get(id=1)
        if not project or not labelstudio_client:
            return {"error": "Label Studio project not initialized"}
        
        # Locate the audio file
        audio_path = "/data/local-files/?d=" + find_ogg_file_path(filename=filename)
        if not audio_path:
            return {
                "success": False,
                "error": f"Audio file {filename} not found in the dataset."
            }

        # Create a task 
        task_data = {
            "data": {
                "audio": audio_path,
                "filename": filename
            }
        }
        task = labelstudio_client.tasks.create(
            project=project.id,
            **task_data
        )
        
        # Get the label interface to create proper prediction format
        li = project.get_label_interface()
        
        # Create predicted label using the control tag name (should match your label config)
        predicted_label = li.get_control('label').label(
            choices=[prediction],
            start=0.0,
            end=100.0,
            labels=[prediction],
            score=round(confidence, 2)
        )
        
        # Create prediction using the SDK's PredictionValue
        prediction_value = PredictionValue(
            model_version="1.0.0",
            result=[predicted_label],
            score=round(confidence, 2)
        )
        
        # Add prediction to the task
        prediction_response = labelstudio_client.predictions.create(
            task=task.id,
            **prediction_value.model_dump()
        )
        
        logger.info(f"Successfully created task and prediction in Label Studio. Task ID: {task.id}, Prediction ID: {prediction_response.id}")
        
        return {
            "success": True,
            "task_id": task.id,
            "prediction_id": prediction_response.id,
            "filename": filename,
            "prediction": prediction,
            "confidence": confidence
        }

    except Exception as e:
        logger.error(f"Error sending prediction to Label Studio: {e}")
        return {
            "success": False,
            "error": str(e),
            "filename": filename,
            "prediction": prediction,
            "confidence": confidence
        }
