# app/router/endpoints.py
import logging
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ml_owls.utils.audio import preprocess_audio
from ml_owls.model.onnx_model import predict
import io
import requests

router = APIRouter()
logger = logging.getLogger(__name__)

# Config & model will be initialized externally
onnx_session = None
label_map = {}
sample_rate = 32000
labelstudio_url = ""
labelstudio_token = ""

@router.get("/health")
def health_check():
    return {"status": "OK"}

@router.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        audio = preprocess_audio(io.BytesIO(contents), sample_rate)
        label, confidence = predict(onnx_session, audio, label_map)
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
