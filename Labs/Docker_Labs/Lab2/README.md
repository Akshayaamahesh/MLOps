# Lab 2: Wine Cultivar Classifier with Docker

This lab trains a machine learning model and serves it as a web app using Docker. It covers two ways to build the same pipeline:

1. **A multi-stage Dockerfile**: one stage trains the model and a second, clean stage serves it.
2. **Docker Compose**: two separate containers that pass the trained model through a shared volume.

The original lab used a TensorFlow neural network on the Iris dataset. This version uses a **Random Forest** on the **Wine** dataset instead.

## Dataset

The [Wine dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-recognition-dataset) ships with scikit-learn. It has 178 wines from three cultivars (grape varieties) grown in the same region of Italy. Each wine has 13 chemical measurements:

alcohol, malic acid, ash, alcalinity of ash, magnesium, total phenols, flavanoids, nonflavanoid phenols, proanthocyanins, color intensity, hue, OD280/OD315 of diluted wines, proline

The goal is to predict the cultivar: **Class 0**, **Class 1** or **Class 2**.

## Model

| Setting | Value |
|---|---|
| Algorithm | `RandomForestClassifier` (scikit-learn) |
| Trees | 200 |
| Max depth | 6 |
| Train/test split | 80/20, stratified, `random_state=42` |
| Test accuracy | 1.00 (36 test samples) |

Random Forests split on one feature threshold at a time, so the inputs don't need scaling. The model is saved with `joblib` as `wine_model.joblib`, together with the feature names, class names and an example input. The web app uses that example to pre-fill its form.

## Project structure

```
Lab2/
├── dockerfile              # Multi-stage build: train, then serve
├── docker-compose.yml      # Two services sharing a volume
├── requirements.txt        # scikit-learn, numpy, joblib, Flask (pinned)
├── HOWTO                   # Quick command reference
└── src/
    ├── model_training.py   # Trains and saves the model
    ├── main.py             # Flask app that serves predictions
    └── templates/
        └── predict.html    # Web form and result view
```

## Prerequisites

- [Docker Desktop](https://www.docker.com/get-started) (includes Docker Compose)
- Port 4000 free on your machine

## Option 1: Multi-stage Dockerfile

```bash
cd Labs/Docker_Labs/Lab2
docker build -t wine-app -f dockerfile .
docker run --rm -p 4000:4000 wine-app
```

How it works:

- **Stage 1 (`model_training`)** installs the requirements and runs `model_training.py`. The training output (accuracy and classification report) appears in the build log.
- **Stage 2 (`serving`)** starts from a fresh `python:3.10-slim` image. It copies in only the trained `wine_model.joblib`, `main.py` and the templates. The training script and anything else from stage 1 stay out of the final image.

Press `Ctrl+C` to stop.

## Option 2: Docker Compose

Stop Option 1 first, because both options use port 4000.

```bash
cd Labs/Docker_Labs/Lab2
docker compose up
```

How it works:

1. `model-training` (container `wine_trainer`) trains the model and copies `wine_model.joblib` into the named volume `model_exchange`. Then it exits.
2. `serving` (container `wine_serving`) waits for training to finish successfully (`depends_on: condition: service_completed_successfully`). It copies the model out of the volume and starts Flask.

To inspect the setup:

```bash
docker compose ps -a     # wine_trainer: Exited (0), wine_serving: Up
docker volume ls         # lab2_model_exchange
```

To clean up:

```bash
docker compose down -v   # -v also removes the shared volume
```

## Using the app

Open **http://localhost:4000/predict** in your browser. The form is pre-filled with a sample wine. Change any of the 13 values and click **Analyze**. The page shows the predicted cultivar and a probability bar for each class.

### API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health message |
| GET | `/predict` | Web form |
| POST | `/predict` | Form-encoded features. Returns the prediction as JSON |

Example request:

```bash
curl -X POST http://localhost:4000/predict \
  -d "alcohol=13.2&malic_acid=1.78&ash=2.14&alcalinity_of_ash=11.2&magnesium=100" \
  -d "total_phenols=2.65&flavanoids=2.76&nonflavanoid_phenols=0.26&proanthocyanins=1.28" \
  -d "color_intensity=4.38&hue=1.05&od280/od315_of_diluted_wines=3.4&proline=1050"
```

Example response:

```json
{
  "predicted_class": "Class 0",
  "probabilities": {"Class 0": 0.98, "Class 1": 0.01, "Class 2": 0.01}
}
```

If a feature is missing or isn't a number, the API returns HTTP 400 with an `error` message.

## Changes from the original lab

- **Dataset:** Iris (4 features) changed to Wine (13 features).
- **Model:** TensorFlow/Keras neural network changed to a scikit-learn Random Forest. TensorFlow is no longer installed, so builds are much faster and the images are much smaller.
- **Ports:** The original Dockerfile and compose file published port 80, but Flask listened on 4000, so the app couldn't be reached. Everything now uses 4000.
- **Output:** The API returns a probability for each class, not just the predicted label.
- **Base image:** `python:3.10` changed to `python:3.10-slim`, with pinned dependency versions.
