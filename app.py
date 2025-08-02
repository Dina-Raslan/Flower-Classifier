import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image

# Load the trained model
model = load_model("flower_model.keras")

# Class names (based on folder order during training)
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# App interface
st.title("🌸 Flower Classifier")
st.write("Upload a flower image and the model will predict its type.")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_container_width=True)

    # Prepare the image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    # Display result
    st.success(f"🌼 Predicted class: **{predicted_class}**")
