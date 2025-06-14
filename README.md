# 🦉 MLOWLS Deployment

Deployment pipeline for the **MLOWLS** inference API designed for the **BirdCLEF 2025 Kaggle Competition**, developed within the **Machine Learning in Data Operations 2025** course at SUPSI/ZHAW.

---

## 📚 Overview

This deployment integrates:

- 🌐 **FastAPI-based inference API**
- 📊 **MLflow** for experiment tracking
- 🔖 **Label Studio** for data labeling
- 📦 **Data Version Control (DVC)** for data and models versioning
- 🐳 **Docker** for easy deployment

---

## 🚀 Quick Start

### Clone and setup environment:

```shell
git clone https://github.com/Manuel-Ippolito/MLOWLS_Deployment.git
cd MLOWLS_Deployment

conda create -n ml_owls python=3.12 -y
conda activate ml_owls
```
make install
🎯 Running the API
Start the FastAPI inference service locally:

```shell
uvicorn ml_owls.main:app --host 0.0.0.0 --port 8000
```
🐳 Docker Deployment
Run all integrated services via Docker Compose:

```shell
docker-compose up --build -d
Service URLs:
FastAPI API: http://localhost:8000
MLflow UI: http://localhost:5000
Label Studio: http://localhost:8080
```
🔮 Prediction API Usage
Example inference request:
```
curl -X POST "http://localhost:8000/predict" \
     -F "audio_file=@path/to/audio.ogg"
```
Returns predicted bird species with confidence scores.

```shell
📂 Project Structure

MLOWLS_Deployment/
├── ml_owls/            # FastAPI inference API
├── mlflow/             # MLflow setup
├── labelstudio/        # Label Studio setup
├── dvc/                # Data Version Control setup
├── docker-compose.yml  # Docker Compose orchestration
├── Makefile            # Automation scripts
└── README.md           # Documentation
```

📈 Experiment Tracking
Access MLflow UI:

```
http://localhost:5000
```

🛠️ Contributions

Contributions welcome. Please submit clear and descriptive Pull Requests.

📄 License

MIT License.

🙏 Acknowledgments
SUPSI/ZHAW: 
- Machine Learning in Data Operations 2025
- BirdCLEF 2025 Kaggle Competition Dataset
