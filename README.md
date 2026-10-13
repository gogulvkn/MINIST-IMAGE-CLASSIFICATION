# MNIST Digit Recognizer

Draw a digit in the browser and get a live prediction from a Keras CNN (~98% test accuracy).

## Stack
- Model: Keras CNN trained on MNIST
- Backend: FastAPI + Uvicorn
- Frontend: HTML canvas + JavaScript

## Structure
main.py             API and preprocessing
index.html          drawing page
mnist_model.keras   trained model
requirements.txt    dependencies

## Run
pip install -r requirements.txt
uvicorn main:app --port 8000

Open http://localhost:8000

## API
POST /predict (image file) returns {"digit": 7, "confidence": 0.99}

## Notes
Input is cropped, scaled to 20x20, and centered in a 28x28 image to match MNIST.