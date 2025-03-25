import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# --------------------------
# Load models
# --------------------------
@st.cache_resource
def load_models():
    return (
        tf.keras.models.load_model("dog_breed_classifier_final.keras"),
        tf.keras.models.load_model("dog_age_classifier.keras"),
    )

breed_model, age_model = load_models()

# --------------------------
# Load class index mappings
# --------------------------
with open("breed_class_indices.json") as f:
    breed_class_indices = {v: k for k, v in json.load(f).items()} 

with open("age_class_indices.json") as f:
    age_index_to_label = {v: k for k, v in json.load(f).items()} 

# --------------------------
# Image preprocessing
# --------------------------
def preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img = np.array(img) / 255.0  # Normalize as done during training
    return np.expand_dims(img, axis=0)

# --------------------------
# Prediction functions
# --------------------------
def predict_breed(img_array):
    preds = breed_model.predict(img_array)[0]
    idx = np.argmax(preds)
    return breed_class_indices.get(idx, "Unknown"), preds[idx]

def predict_age(img_array):
    preds = age_model.predict(img_array)[0]
    idx = np.argmax(preds)
    return age_index_to_label.get(idx, "Unknown"), preds[idx]

# --------------------------
# Streamlit UI
# --------------------------
st.title("🐶 Dog Breed & Age Classifier")

option = st.radio("Choose a prediction type:", ["🐕 Dog Breed", "🧓 Dog Age Group"])
uploaded_file = st.file_uploader("Upload a dog image", type=["jpg", "jpeg", "png"])

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
