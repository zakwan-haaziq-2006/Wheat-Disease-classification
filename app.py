import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# ----------------------------
# Page configuration
# ----------------------------
st.set_page_config(
    page_title='🌱 Wheat Disease Detector',
    layout='centered',
    initial_sidebar_state='expanded'
)

# ----------------------------
# Page header
# ----------------------------
st.markdown("""
    <div style='text-align: center;'>
        <h1>🌱 Wheat Disease Detector</h1>
        <p>Upload an image of a wheat leaf to detect possible diseases.</p>
    </div>
""", unsafe_allow_html=True)

st.write("---")  # horizontal separator

# ----------------------------
# File upload section
# ----------------------------
uploaded_file = st.file_uploader(
    label="Choose a wheat leaf image",
    type=["jpg", "jpeg", "png"]
)

# ----------------------------
# Load model once
# ----------------------------
model = load_model('Mobile_net_model.h5')  # replace with your model path

if uploaded_file is not None:
    # Resize uploaded image to model's input size
    image = Image.open(uploaded_file)
    image = image.resize((256, 256))  # match your CNN input size
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Predict button
    if st.button("Predict"):
        # Preprocess image
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # normalize

        # Prediction
        pred_probs = model.predict(img_array)
        class_idx = np.argmax(pred_probs, axis=1)[0]

        # Map class index to class name
        class_labels = ['Aphid','Black Rust','Blast','Brown Rust','Common Root Rot','Fusarium Head Blight','Healthy','Leaf Blight','Mildew','Mite','Septoria','Smut','Stem fly','Tan spot','Yellow Rust']  # update according to your dataset
        predicted_class = class_labels[class_idx]

        # Display predicted class
        st.markdown(f"<h2 style='text-align: center; color: green;'>Prediction: {predicted_class}</h2>", unsafe_allow_html=True)
