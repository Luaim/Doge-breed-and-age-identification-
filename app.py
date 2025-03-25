import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image

# --------------------------
# Utility Function to Get Absolute Paths
# --------------------------
def get_absolute_path(filename):
    """Returns the absolute path of a file in the same directory as this script."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

# --------------------------
# Load Models
# --------------------------
@st.cache_resource
def load_models():
    """Loads the pre-trained models for dog breed and age classification."""
    
    breed_model_path = get_absolute_path("dog_breed_classifier_final.keras")
    age_model_path = get_absolute_path("dog_age_classifier.keras")

    # Debugging: Print paths to ensure correctness
    st.write(f"🔍 Checking Model Paths:\n- Breed Model: {breed_model_path}\n- Age Model: {age_model_path}")

    # Ensure models exist before loading
    if not os.path.exists(breed_model_path):
        st.error(f"❌ Model file not found: {breed_model_path}")
        raise FileNotFoundError(f"File not found: {breed_model_path}")
    
    if not os.path.exists(age_model_path):
        st.error(f"❌ Model file not found: {age_model_path}")
        raise FileNotFoundError(f"File not found: {age_model_path}")

    # Load models
    breed_model = tf.keras.models.load_model(breed_model_path)
    age_model = tf.keras.models.load_model(age_model_path)
    
    return breed_model, age_model

breed_model, age_model = load_models()

# --------------------------
# Load Class Index Mappings
# --------------------------
with open(get_absolute_path("breed_class_indices.json")) as f:
    class_to_index_breed = json.load(f)
breed_class_indices = {v: k for k, v in class_to_index_breed.items()}  # Corrected: {0: "Chihuahua"}

with open(get_absolute_path("age_class_indices.json")) as f:
    class_to_index_age = json.load(f)
age_index_to_label = {v: k for k, v in class_to_index_age.items()}  # Corrected: {0: "Adult", 1: "Senior", ...}

# --------------------------
# Image Preprocessing
# --------------------------
def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model input."""
    img = image.resize(target_size)
    img = np.array(img) / 255.0  # Normalize
    return np.expand_dims(img, axis=0)

# --------------------------
# Prediction Functions
# --------------------------
def predict_breed(img_array):
    """Predict dog breed from image array."""
    preds = breed_model.predict(img_array)[0]
    idx = np.argmax(preds)
    label = breed_class_indices.get(idx, "Unknown")
    confidence = preds[idx]
    return label, confidence

def predict_age(img_array):
    """Predict dog age group from image array."""
    preds = age_model.predict(img_array)[0]
    idx = np.argmax(preds)
    label = age_index_to_label.get(idx, "Unknown")
    confidence = preds[idx]
    return label, confidence

# --------------------------
# Streamlit UI
# --------------------------
st.title("🐶 Dog Breed & Age Classifier")

# Debugging: Show files in the working directory
st.write("📂 Current Directory:", os.getcwd())
st.write("📄 Files in Directory:", os.listdir(os.getcwd()))

option = st.radio("Choose a prediction type:", ["🐕 Dog Breed", "🧓 Dog Age Group"])

uploaded_file = st.file_uploader("📸 Upload a dog image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = preprocess_image(image)

    if option == "🐕 Dog Breed":
        with st.spinner("Predicting breed..."):
            breed, confidence = predict_breed(img_array)
        st.success(f"**Breed:** {breed} ({confidence * 100:.2f}%)")

    elif option == "🧓 Dog Age Group":
        with st.spinner("Predicting age group..."):
            age, confidence = predict_age(img_array)
        st.success(f"**Age Group:** {age} ({confidence * 100:.2f}%)")
