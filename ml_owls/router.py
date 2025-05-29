import logging
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ml_owls.model.onnx_model import predict
import requests

router = APIRouter()
logger = logging.getLogger(__name__)

onnx_session = None
label_map = {}
labelstudio_url = ""
labelstudio_token = ""

@router.get("/health")
def health_check():
    return {"status": "OK"}

@router.get("/readiness")
def readiness_check():
    if onnx_session is None or not label_map:
        logger.warning("Service not ready: model or label map not initialized.")
        return JSONResponse(status_code=503, content={"status": "not ready"})
    return {"status": "ready"}

@router.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        label, confidence = predict(onnx_session, contents, label_map)
        return {"prediction": label, "confidence": confidence}
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@router.post("/label")
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
