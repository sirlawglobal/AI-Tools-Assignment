import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load pre-trained MNIST model
model = tf.keras.models.load_model('mnist_cnn.h5')  # Save your trained model first

st.title("MNIST Handwritten Digit Classifier")

# Upload image
uploaded_file = st.file_uploader("Upload a handwritten digit image (28x28, grayscale)", type=["png", "jpg"])
if uploaded_file:
    # Preprocess image
    img = Image.open(uploaded_file).convert('L').resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    # Display results
    st.image(img, caption="Uploaded Image", width=100)
    st.write(f"Predicted Digit: {predicted_digit}")

# Save screenshot for report
st.write("Save this page as a screenshot for the report!")
