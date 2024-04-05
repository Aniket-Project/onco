import streamlit as st
import tensorflow as tf
import numpy as np
import pickle

# Load the model data from the pickle file
with open('model_data_2.pkl', 'rb') as file:
    pickle_data = pickle.load(file)

# Extract model and class names from the pickle data
model = pickle_data['model']
class_names = pickle_data['class_names']

# Define function to preprocess and predict image
def predict(image):
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch of images

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)

    return predicted_class, confidence

# Streamlit app
def main():
    st.title("Cancer Classification App")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(256, 256))
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Get prediction
        predicted_class, confidence = predict(image)
        st.write("Predicted Class:", predicted_class)
        st.write("Confidence:", confidence, "%")

if __name__ == '__main__':
    main()
