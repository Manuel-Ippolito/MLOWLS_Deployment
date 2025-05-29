# app/labelstudio_integration.py
import requests
import base64
import ml_owls.router as router_module

# Load label studio url from config and token from .env
LABEL_STUDIO_URL = router_module.labelstudio_url
LABEL_STUDIO_TOKEN = router_module.labelstudio_token

def send_to_labelstudio(filename, file_bytes):
    base64_audio = base64.b64encode(file_bytes).decode("utf-8")
    headers = {"Authorization": f"Token {LABEL_STUDIO_TOKEN}"}
    task = {
        "data": {
            "audio": f"data:audio/wav;base64,{base64_audio}",
            "filename": filename
        }
    }
    response = requests.post(LABEL_STUDIO_URL, headers=headers, json=[task])
    response.raise_for_status()
    return response.json()[0].get("id", "unknown")
