import streamlit as st
import streamlit_drawable_canvas as st_canvas
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import cv2

# Load the trained MNIST digit recognition model
model = load_model("mnist_digit_recognition_model.h5")


# Define a function to preprocess and classify the user's drawing
def classify_digit(drawing_image):
    # Resize the image to 28x28 pixels
    # drawing_image = ImageOps.fit(drawing_image, (28, 28), Image.ANTIALIAS)

    # Convert the image to grayscale
    # drawing_image = ImageOps.grayscale(drawing_image)

    # Invert the image (MNIST digits are white on a black background)
    # drawing_image = ImageOps.invert(drawing_image)

    # Convert the image to a NumPy array
    drawing_array = np.array(drawing_image).astype("float")

    # Normalize pixel values to be between 0 and 1
    drawing_array = drawing_array / 255.0
    # return(drawing_array.shape)

    # Reshape the image to match the model's input shape (1, 28, 28)
    # drawing_array = np.reshape(drawing_array, (1, 28, 28, 1))  # Add the channel dimension

    # Reshape the image to match the model's input shape (1, 28, 28)
    drawing_array = np.reshape(drawing_array, (1, 28, 28))

    # Predict the digit using the loaded model
    digit_prediction = model.predict(drawing_array)

    # Get the predicted digit (index of the maximum probability)
    predicted_digit = np.argmax(digit_prediction)

    return predicted_digit


# Streamlit UI elements
st.title("MNIST Digit Recognition")

st.sidebar.markdown("Draw a digit from 0 to 9 on the canvas below:")
canvas = st_canvas.st_canvas(
    stroke_color="white", width=200, height=200, drawing_mode="freedraw"
)  # 加上stroke_color='white'

if st.button("Classify"):
    # Get the user's drawing and classify it
    user_drawing = cv2.cvtColor(canvas.image_data, cv2.COLOR_RGB2GRAY)  # 加上色彩空間轉換
    user_drawing = cv2.resize(user_drawing, (28, 28))  # 加上尺寸轉換
    cv2.imwrite("test.jpg", user_drawing)
    user_digit = classify_digit(user_drawing)

    st.write(f"Predicted Digit: {user_digit}")

st.sidebar.text("Note: Click the 'Classify' button after drawing.")
