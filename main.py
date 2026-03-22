import tensorflow as tf
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageFilter
import io
import uvicorn

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1️⃣ Load TFLite
interpreter = tf.lite.Interpreter(model_path="corn_doctor_float16.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("✅ TFLite model loaded!")

class_names = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy", "Other"]

# 2️⃣ Texture checker
def has_leaf_texture(img):
    gray = img.convert('L').resize((224, 224))
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_np = np.array(edges)
    return (edge_np > 20).sum() / edge_np.size

# 3️⃣ Predict endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Texture check
        edge_density = has_leaf_texture(image)
        if edge_density < 0.05:
            return {
                "status": "REJECTED",
                "prediction": "Other",
                "confidence": 0.0,
                "message": "🚫 Hindi ito dahon ng mais. Pakikuha muli ng maayos na larawan."
            }

        # Preprocess for TFLite
        img = image.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array -= [0.485, 0.456, 0.406]
        img_array /= [0.229, 0.224, 0.225]
        img_array = np.expand_dims(img_array, axis=0)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]

        # Softmax
        probs = np.exp(output) / np.sum(np.exp(output))
        pred_idx = np.argmax(probs)
        confidence = float(probs[pred_idx])
        pred_class = class_names[pred_idx]

        if pred_class == "Other":
            return {
                "status": "REJECTED",
                "prediction": "Other",
                "confidence": confidence,
                "message": "🚫 Hindi ito dahon ng mais. Pakikuha muli ng maayos na larawan."
            }
        elif confidence >= 0.90:
            return {
                "status": "SUCCESS",
                "prediction": pred_class,
                "confidence": confidence,
                "message": f"Ang resulta ay {pred_class}. Maaari mo nang makita ang lunas."
            }
        elif confidence >= 0.70:
            return {
                "status": "REPORT",
                "prediction": pred_class,
                "confidence": confidence,
                "message": "⚠️ Posibleng may sakit. Ang ulat ay ipapadala sa CASD para sa kumpirmasyon."
            }
        else:
            return {
                "status": "UNCERTAIN",
                "prediction": pred_class,
                "confidence": confidence,
                "message": "❓ Hindi sigurado ang AI. Pakikuha muli ng mas malinaw na larawan."
            }

    except Exception as e:
        return {"status": "ERROR", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)