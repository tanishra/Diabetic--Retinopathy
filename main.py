import pickle
import streamlit as st
import numpy as np
from PIL import Image

# Load the trained model
model_path = 'model.pkl'

# Function to load the model
def load_model(model_path):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
            return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Load the model
model = load_model(model_path)

# Check if the model is valid
if not hasattr(model, 'predict'):
    st.error("The loaded model does not have a 'predict' method. Please check the model file.")
    st.stop()

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
    img_array = np.array(img)        # Convert the image to a numpy array
    img_array = img_array / 255.0    # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Ensure the image array has the correct shape
    if img_array.shape != (1, 224, 224, 3):
        st.error(f"Invalid input shape: {img_array.shape}. Expected shape: (1, 224, 224, 3).")
    else:
        # Make prediction
        if st.button("Predict"):
            try:
                prediction = model.predict(img_array)  # Get raw predictions
                predicted_class = np.argmax(prediction, axis=1)[0]  # Get the index of the class with the highest probability
                
                # Display the prediction
                st.write(f"Prediction: {labels_dict[predicted_class]}")

                # Show confidence scores
                st.write("Confidence scores:")
                for i, score in enumerate(prediction[0]):
                    st.write(f"{labels_dict[i]}: {score * 100:.2f}%")

            except Exception as e:
                st.error(f"Error making prediction: {e}")
