import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

app = FastAPI()

# Only needed if you open index.html from disk instead of via the server
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

model = load_model("mnist_model.keras")


def preprocess(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    arr = np.array(img)

    # Crop to the digit's bounding box
    ys, xs = np.where(arr > 30)
    if len(xs) == 0:
        return np.zeros((1, 28, 28, 1), dtype="float32")
    arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # Scale longest side to 20px, keep aspect ratio
    h, w = arr.shape
    scale = 20.0 / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    digit = Image.fromarray(arr).resize((new_w, new_h), Image.LANCZOS)

    # Paste into 28x28 canvas
    canvas = np.zeros((28, 28), dtype="float32")
    top, left = (28 - new_h) // 2, (28 - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = np.array(digit)

    # Shift so center of mass is at the center (as in MNIST)
    total = canvas.sum()
    if total > 0:
        cy = (canvas.sum(axis=1) * np.arange(28)).sum() / total
        cx = (canvas.sum(axis=0) * np.arange(28)).sum() / total
        canvas = np.roll(canvas, (round(14 - cy), round(14 - cx)), axis=(0, 1))

    return (canvas / 255.0).reshape(1, 28, 28, 1)





@app.get("/")
def root():
    return FileResponse("index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    x = preprocess(await file.read())
    probs = model.predict(x, verbose=0)[0]
    return {
        "digit": int(np.argmax(probs)),
        "confidence": float(np.max(probs)),
    }