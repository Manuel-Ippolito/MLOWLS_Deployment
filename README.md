# ML Owls Project

This repository contains the MLOps pipeline for Animal Species Identification from Audio.
The project was made for the BirdCLEF-2025 Kaggle competition and for the ""Machine Learning in Data 2025"" course at SUPSI.


## Install project and dependencies:

Create a new Python 3.12 environment:

```shell
conda create --name ml_owls python=3.12
```

Activate the conda environment just created:

```shell
conda activate ml_owls
```

Install dependencies on the environment from the pyproject.toml:

```shell
pip install --no-cache-dir -e ".[inference]"
```

## Deploy the whole project as a single Docker Container:

Very simply deploy the project in a container using one command:
```shell
docker compose up -d
```

## Run the services singularly:

API for Model inference:

```shell
uvicorn ml_owls.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Label Studio for Data Annotation:

```shell
label-studio start --host 0.0.0.0 --port 8080 --data-dir /data"
```

MLFlow for model monitoring and experiment tracking:

```shell
mlflow server --backend-store-uri sqlite:////mlflow/mlflow.db --default-artifact-root /mlflow/mlruns --host 0.0.0.0 --port 5000
```

To get the data we used, please request us access and we'll provide indications to get it via DVC.
Once you ahve received the necessary permissions and files, you can run:

```shell
dvc pull --verbose
```
