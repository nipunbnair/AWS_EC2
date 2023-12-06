import streamlit as st
from keras.models import load_model
import cv2
import numpy as np


def classify_file(file):
    model = load_model('my_model.h5')
    model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
    img = cv2.imread(file)
    img = cv2.resize(img,(320,240))
    img = np.reshape(img,[1,320,240,3])
    classes = model.predict_classes(img)
    print(classes)
    return str(classes)

def main():
    st.title("File Upload and Classification")

    uploaded_file = st.file_uploader("Select Image:", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Perform classification on file upload
        classifier_response = classify_file(uploaded_file)

        # Display classification result
        st.subheader("Classification Result")
        st.write(f"Classification Result: {classifier_response}")

        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

if __name__ == "__main__":
    main()
