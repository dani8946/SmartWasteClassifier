import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Smart Waste Classifier",
    page_icon="🧠",
    layout="centered"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
    <style>
        .stApp {
            background-color: #fffbea;
        }
        .title {
            font-size: 40px;
            color: #fdd835;
            text-align: center;
            font-weight: bold;
            padding: 20px 0;
        }
        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #333333;
        }
        .prediction-box {
            background-color: #666666;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        }
        .footer {
            font-size: 13px;
            color: #fffde7;
            text-align: center;
            margin-top: 30px;
        }
    </style>
""", unsafe_allow_html=True)

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mobilenetv2_trashnet.h5")
    return model

model = load_model()
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# ==================== PAGE CONTENT ====================
st.markdown("<div class='title'>♻️ Smart Waste Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload an image of waste and get its type with confidence score</div>", unsafe_allow_html=True)
st.markdown("---")

uploaded_file = st.file_uploader("📤 Upload a trash image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='🖼️ Uploaded Image', use_column_width=True)

    # === Preprocess Image ===
    img = image.resize((224, 224))
    img_array = np.array(img)
    if img_array.shape[-1] == 4:  # RGBA to RGB
        img_array = img_array[..., :3]
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # === Prediction ===
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = prediction[0][predicted_index] * 100

    # === Result Display ===
    st.markdown("---")
    st.markdown(f"<div class='prediction-box'>"
                f"<h3>🧠 Prediction: <span style='color:#1976D2'>{predicted_class.capitalize()}</span></h3>"
                f"<p>🔍 Confidence: <b>{confidence:.2f}%</b></p>"
                f"</div>", unsafe_allow_html=True)

    # === Class Probabilities ===
    with st.expander("📊 See class probabilities"):
        for i, prob in enumerate(prediction[0]):
            st.write(f"**{class_names[i].capitalize()}**: {prob*100:.2f}%")

# ==================== FOOTER ====================
st.markdown("<div class='footer'>© 2025 Smart Waste Classification | Made with 💙 using Streamlit</div>", unsafe_allow_html=True)
