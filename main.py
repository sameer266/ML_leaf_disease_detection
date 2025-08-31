from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load your trained model
model = load_model("leaf_cnn_model.h5")

app = FastAPI(title="Leaf Disease Prediction API")

# Define class names
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

disease_mapping = {
    'Apple___Apple_scab': "Apple Leaf - Unhealthy (Disease: Apple scab)",
    'Apple___Black_rot': "Apple Leaf - Unhealthy (Disease: Black rot)",
    'Apple___Cedar_apple_rust': "Apple Leaf - Unhealthy (Disease: Cedar apple rust)",
    'Apple___healthy': "Apple Leaf - Healthy",

    'Blueberry___healthy': "Blueberry Leaf - Healthy",

    'Cherry_(including_sour)___Powdery_mildew': "Cherry (including sour) Leaf - Unhealthy (Disease: Powdery mildew)",
    'Cherry_(including_sour)___healthy': "Cherry (including sour) Leaf - Healthy",

    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "Corn (maize) Leaf - Unhealthy (Disease: Cercospora leaf spot Gray leaf spot)",
    'Corn_(maize)___Common_rust_': "Corn (maize) Leaf - Unhealthy (Disease: Common rust)",
    'Corn_(maize)___Northern_Leaf_Blight': "Corn (maize) Leaf - Unhealthy (Disease: Northern Leaf Blight)",
    'Corn_(maize)___healthy': "Corn (maize) Leaf - Healthy",

    'Grape___Black_rot': "Grape Leaf - Unhealthy (Disease: Black rot)",
    'Grape___Esca_(Black_Measles)': "Grape Leaf - Unhealthy (Disease: Esca (Black Measles))",
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': "Grape Leaf - Unhealthy (Disease: Leaf blight (Isariopsis Leaf Spot))",
    'Grape___healthy': "Grape Leaf - Healthy",

    'Orange___Haunglongbing_(Citrus_greening)': "Orange Leaf - Unhealthy (Disease: Haunglongbing (Citrus greening))",

    'Peach___Bacterial_spot': "Peach Leaf - Unhealthy (Disease: Bacterial spot)",
    'Peach___healthy': "Peach Leaf - Healthy",

    'Pepper,_bell___Bacterial_spot': "Pepper (bell) Leaf - Unhealthy (Disease: Bacterial spot)",
    'Pepper,_bell___healthy': "Pepper (bell) Leaf - Healthy",

    'Potato___Early_blight': "Potato Leaf - Unhealthy (Disease: Early blight)",
    'Potato___Late_blight': "Potato Leaf - Unhealthy (Disease: Late blight)",
    'Potato___healthy': "Potato Leaf - Healthy",

    'Raspberry___healthy': "Raspberry Leaf - Healthy",

    'Soybean___healthy': "Soybean Leaf - Healthy",

    'Squash___Powdery_mildew': "Squash Leaf - Unhealthy (Disease: Powdery mildew)",

    'Strawberry___Leaf_scorch': "Strawberry Leaf - Unhealthy (Disease: Leaf scorch)",
    'Strawberry___healthy': "Strawberry Leaf - Healthy",

    'Tomato___Bacterial_spot': "Tomato Leaf - Unhealthy (Disease: Bacterial spot)",
    'Tomato___Early_blight': "Tomato Leaf - Unhealthy (Disease: Early blight)",
    'Tomato___Late_blight': "Tomato Leaf - Unhealthy (Disease: Late blight)",
    'Tomato___Leaf_Mold': "Tomato Leaf - Unhealthy (Disease: Leaf Mold)",
    'Tomato___Septoria_leaf_spot': "Tomato Leaf - Unhealthy (Disease: Septoria leaf spot)",
    'Tomato___Spider_mites Two-spotted_spider_mite': "Tomato Leaf - Unhealthy (Disease: Spider mites Two-spotted spider mite)",
    'Tomato___Target_Spot': "Tomato Leaf - Unhealthy (Disease: Target Spot)",
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "Tomato Leaf - Unhealthy (Disease: Tomato Yellow Leaf Curl Virus)",
    'Tomato___Tomato_mosaic_virus': "Tomato Leaf - Unhealthy (Disease: Tomato mosaic virus)",
    'Tomato___healthy': "Tomato Leaf - Healthy"
}

# Helper function to process image
def process_image(img):
    IMG_SIZE = (64, 64)
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array

# API route
# API route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img = Image.open(file.file).convert("RGB")
        img_array = process_image(img)
        predictions = model.predict(img_array)
        
        # Get model prediction
        predicted_class = class_names[np.argmax(predictions)]
        confidence = float(np.max(predictions))
        
        # Convert to user-friendly response
        user_friendly = disease_mapping.get(predicted_class, predicted_class)
        
        return JSONResponse({
            "status": "success",
            "prediction": user_friendly,
            "confidence": round(confidence, 3)
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        })