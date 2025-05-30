from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests
from fastapi import Response
import os
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","https://potato-disease-classification-frontend.onrender.com","https://potato-disease-classification-vapb.onrender.com"],  # your React origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MODEL = tf.keras.models.load_model("../saved_models/1.keras")
endpoint = "http://localhost:8503/v1/models/tomatoes_model:predict"
CLASS_NAMES = ['Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_healthy']

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.head("/", include_in_schema=False)
async def head_root():
    return Response(status_code=200)

@app.get("/", include_in_schema=False)
async def health_check():
    return {"status": "ok", "service": "Tomato Disease Classifcation API"}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image,0)
    json_data = {
        "instances":img_batch.tolist()
    }
    response = requests.post(endpoint,json=json_data)
    prediction = np.array(response.json()["predictions"][0])
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)
    return {
        "class":predicted_class,
        "confidence":float(confidence)
    }
    

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )