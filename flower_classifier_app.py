import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Load the pre-trained model
model = load_model('C:/Users/Youssef/Desktop/project/flower_classification_model.h5')

# Define class labels
class_labels = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

def classify_image(image):
    # Load and preprocess the image
    img = load_img(image, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values

    # Predict the class probabilities
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])

    return class_labels[predicted_class]

def main():
    st.title("Flower Classification App")
    st.write("Upload an image and let the model classify the flower!")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        prediction = classify_image(uploaded_file)
        st.write(f"Prediction: {prediction}")

if __name__ == '__main__':
    main()
