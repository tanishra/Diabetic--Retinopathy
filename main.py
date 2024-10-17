import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Define a dictionary to map label indices to categories
labels_dict = {
    0: 'No Diabetic Retinopathy',
    1: 'Mild Diabetic Retinopathy',
    2: 'Moderate Diabetic Retinopathy',
    3: 'Severe Diabetic Retinopathy',
    4: 'Proliferative Diabetic Retinopathy'
}

# Streamlit UI
st.title("Diabetic Retinopathy Detection")

st.write("Upload a retina image to detect diabetic retinopathy.")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Retina Image', use_column_width=True)
    
    # Preprocess the image
    img = image.resize((224, 224))  # Resize the image to match model input size
    img_array = np.array(img)       # Convert the image to a numpy array
    img_array = img_array / 255.0   # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    # Display the prediction
    st.write(f"Prediction: {labels_dict[predicted_class]}")

    # Show confidence scores
    st.write("Confidence scores:")
    for i, score in enumerate(prediction[0]):
        st.write(f"{labels_dict[i]}: {score*100:.2f}%")

