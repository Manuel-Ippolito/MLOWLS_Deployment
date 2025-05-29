# MLOWLS_Deployment
This repository contains the deployment pipeline for project made for the BirdCLEF-2025 Kaggle competition and for the Machine Learning in Data 2025 course at SUPSI


## Install project and dependencies:

Create a new Python 3.12 environment:
```shell
conda create --name ml_owls python=3.12
```

Activate the conda environment just created:
```shell
conda activate ml_owls
```

Install dependencies on the environment:
```shell
make install
```

## Use the MLOwls API service"

Run it locally:
```shell
uvicorn ml_owls.main:app", "--host", "0.0.0.0", "--port", "8000"
```