# app/model/onnx_model.py
import onnxruntime as ort
import numpy as np
import logging

logger = logging.getLogger(__name__)

def load_model(model_path: str):
    logger.info(f"Loading ONNX model from {model_path}")
    return ort.InferenceSession(model_path)

def predict(session, input_tensor: np.ndarray, label_map: dict):
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_tensor})
    probs = output[0][0]
    pred_index = int(np.argmax(probs))
    confidence = float(probs[pred_index])
    return label_map.get(str(pred_index), "unknown"), confidence
