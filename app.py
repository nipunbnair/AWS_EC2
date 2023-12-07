import streamlit as st
from tensorflow.keras.models import load_model
import cv2
import numpy as np

def classify_file(file):
  """
  Classifies an image file.

  Args:
    file: Uploaded image file.

  Returns:
    str: The predicted class label.
  """
  # Load the model
  model = load_model('my_model.h5')

  # Read the uploaded image
  try:
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
  except Exception as e:
    print(f"Error reading image: {e}")
    return None

  # Resize the image
  img = cv2.resize(img, (64, 64))

  # Reshape the image for model prediction
  img = np.reshape(img, [1, 64, 64, 3])

  # Predict the class
  classes = np.argmax(model.predict(img))

  # Return the predicted class label
  return str(classes)

def main():
  st.title("File Upload and Classification")

  # Allow uploading images of various formats
  uploaded_file = st.file_uploader("Select Image:", type=["jpg", "jpeg", "png", "bmp"])

  if uploaded_file is not None:
    # Perform classification on uploaded file
    classifier_response = classify_file(uploaded_file)

    # Handle error in case of unsuccessful classification
    if classifier_response is None:
      st.error("Error classifying the image. Please try again.")
      return

    # Display classification result
    st.subheader("Classification Result")
    st.write(f"Classification Result: {classifier_response}")

    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

if __name__ == "__main__":
  main()
