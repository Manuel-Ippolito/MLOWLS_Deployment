import logging
from ml_owls.main import labelstudio_client, project

logger = logging.getLogger(__name__)

def send_to_labelstudio(filename: str, prediction: str, confidence: float):
    """
    Send prediction results to Label Studio.
    
    Args:
        filename (str): Name of the audio file.
        prediction (str): Predicted species name (must match one of the labels in taxonomy).
        confidence (float): Confidence score of the prediction (0.0 to 1.0).
        audio_url (str): URL to the audio file (optional).
  
    Returns:
        dict: Response containing task_id and prediction_id, or error information.
    """
    try:
        if not project:
            return {"error": "Label Studio project not initialized"}
        
        # Create a task first
        task_data = {
            "data": {
                "audio": f"/audio/{filename}",
                "filename": filename
            },
            "predictions": [
                {
                    "model_version": "1.0.0",
                    "score": confidence,
                    "result": [
                        {
                            "from_name": "label",
                            "to_name": "audio",
                            "type": "choices",
                            "value": {
                                "choices": [prediction]
                            }
                        }
                    ]
                }
            ]
        }
        
        # Create the task
        task = labelstudio_client.tasks.create(
            project=project.id,
            **task_data
        )
        
        logger.info(f"Successfully created task with prediction in Label Studio. Task ID: {task.id}")
        
        return {
            "success": True,
            "task_id": task.id,
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
