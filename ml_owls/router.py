import logging
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.params import Depends
from fastapi.responses import JSONResponse
from ml_owls.auth import get_api_key
from ml_owls.model.onnx_model import predict
import requests

router = APIRouter()
logger = logging.getLogger(__name__)

# These should be initialized from main.py
onnx_session = None
labelstudio_url = ""
labelstudio_token = ""

@router.get("/health")
def health_check():
    return {"status": "OK"}

@router.get("/readiness")
def readiness_check():
    if onnx_session is None:
        logger.warning("Service not ready: model or label map not initialized.")
        return JSONResponse(status_code=503, content={"status": "not ready"})
    return {"status": "ready"}

@router.post("/predict", dependencies=[Depends(get_api_key)])
async def predict_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        label, confidence = predict(onnx_session, contents)
        return {"prediction": label, "confidence": confidence}
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")

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