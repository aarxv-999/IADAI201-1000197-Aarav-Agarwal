# app.py
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model once
@st.cache_resource
def load_waste_model():
    model = load_model('waste_classifier_mobilenet.h5')
    return model

model = load_waste_model()
labels = ['battery','biological','cardboard','clothes','glass','metal','paper','plastic','shoes','trash']

label_to_category = {
  'battery': 'hazardous',
  'biological': 'biodegradable',
  'cardboard': 'recyclable',
  'clothes': 'recyclable',
  'glass': 'recyclable',
  'metal': 'recyclable',
  'paper': 'recyclable',
  'plastic': 'recyclable',
  'shoes': 'recyclable',
  'trash': 'biodegradable'
}
category_to_bin = {'biodegradable':'green','recyclable':'blue','hazardous':'red'}

st.set_page_config(page_title="SmartWasteAI", layout="centered")
st.title("♻️ SmartWasteAI – Waste Classification & Bin Recommendation")
st.write("Upload an image of waste to predict its category and recommended bin color.")

uploaded = st.file_uploader("Upload a waste image", type=["jpg","jpeg","png"])

if uploaded is not None:
    img = Image.open(uploaded).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess
    img_resized = img.resize((224,224))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    preds = model.predict(img_array)
    pred_idx = np.argmax(preds)
    pred_label = labels[pred_idx]
    confidence = preds[0][pred_idx] * 100
    category = label_to_category[pred_label]
    bin_color = category_to_bin[category]
    
    st.subheader("Prediction Results")
    st.write(f"**Predicted Waste Type:** {pred_label}")
    st.write(f"**Category:** {category.capitalize()}")
    st.write(f"**Confidence:** {confidence:.2f}%")
    st.markdown(f"<div style='width:120px;height:40px;background:{bin_color};border-radius:10px;text-align:center;line-height:40px;color:white'><b>{bin_color.upper()} BIN</b></div>", unsafe_allow_html=True)

