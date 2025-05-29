# app/labelstudio_integration.py
import requests
import base64

LABEL_STUDIO_URL = "http://labelstudio:8080/api/projects/1/import"
LABEL_STUDIO_TOKEN = "YOUR_TOKEN"

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
