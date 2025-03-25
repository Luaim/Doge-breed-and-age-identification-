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
    breed_model = tf.keras.models.load_model("dog_breed_classifier_final.keras")
    age_model = tf.keras.models.load_model("dog_age_classifier.keras")
    return breed_model, age_model

breed_model, age_model = load_models()

# --------------------------
# Load breed and age class index mappings and health recommendations
# --------------------------
with open("breed_class_indices.json") as f:
    breed_health_data = json.load(f)  # Load breed health data

with open("age_class_indices.json") as f:
    age_health_data = json.load(f)  # Load age group health data

# --------------------------
# Image preprocessing
# --------------------------
def preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img = np.array(img) / 255.0  # Normalization used during training
    return np.expand_dims(img, axis=0)

# --------------------------
# Prediction functions
# --------------------------
def predict_breed(img_array):
    preds = breed_model.predict(img_array)[0]
    idx = np.argmax(preds)
    breed_name = list(breed_health_data.keys())[idx]  # Get breed name from the key
    confidence = preds[idx]
    return breed_name, confidence

def predict_age(img_array):
    preds = age_model.predict(img_array)[0]
    idx = np.argmax(preds)
    age_group = list(age_health_data.keys())[idx]  # Get age group from the key
    confidence = preds[idx]
    return age_group, confidence

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

            # Fetch and display health recommendation for the breed
            if breed in breed_health_data:
                health_info = breed_health_data[breed]["health"]
                st.subheader("Healthcare Recommendation")
                st.write(f"{health_info}")

    elif option == "🧓 Dog Age Group":
        with st.spinner("Predicting age group..."):
            age, confidence = predict_age(img_array)
            st.success(f"**Age Group:** {age} ({confidence * 100:.2f}%)")

            # Fetch and display health recommendation for the age group
            if age in age_health_data:
                age_health_info = age_health_data[age]["health"]
                st.subheader("Healthcare Recommendation")
                st.write(f"{age_health_info}")
