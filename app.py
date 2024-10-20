import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import tensorflow as tf

# Define the updated FixedDropout layer
class FixedDropout(tf.keras.layers.Layer):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(FixedDropout, self).__init__(**kwargs)
        self.rate = rate
        self.noise_shape = noise_shape
        self.seed = seed

    def call(self, inputs, training=None):
        if 0. < self.rate < 1. and training:
            return tf.keras.backend.dropout(inputs, self.rate, noise_shape=self.noise_shape, seed=self.seed)
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(FixedDropout, self).get_config()
        config.update({
            "rate": self.rate,
            "noise_shape": self.noise_shape,
            "seed": self.seed
        })
        return config

# Load the model with custom layers/activation
model = load_model('model.h5', custom_objects={
    'swish': tf.keras.activations.swish,
    'FixedDropout': FixedDropout
})

# Define function to preprocess the image
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize image to the size expected by the model
    img = img.convert('RGB')      # Ensure it's in RGB format
    img_array = image.img_to_array(img)  # Convert the image to a numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to fit model input
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Define function to make predictions
def predict_image(img):
    preprocessed_img = preprocess_image(img)
    prediction = model.predict(preprocessed_img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    # Mapping label numbers to diabetic retinopathy categories
    categories = {
        0: 'No Diabetic Retinopathy',
        1: 'Mild Diabetic Retinopathy',
        2: 'Moderate Diabetic Retinopathy',
        3: 'Severe Diabetic Retinopathy',
        4: 'Proliferative Diabetic Retinopathy'
    }
    return categories[predicted_class]

# Streamlit UI
st.title("Diabetic Retinopathy Detection")

# File uploader widget
uploaded_file = st.file_uploader("Choose a retina image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Retina Image", use_column_width=True)
    
    # Make prediction
    if st.button("Predict"):
        result = predict_image(img)
        st.success(f"Prediction: {result}")

