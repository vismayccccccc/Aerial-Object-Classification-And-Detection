import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("bird_drone_classifier.h5")

class_names = ['bird','drone']

st.title("Bird vs Drone Image Classifier")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    predicted_class = class_names[int(prediction[0] > 0.5)]

    st.subheader("Prediction:")
    st.write(predicted_class)
