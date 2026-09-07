import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tflite
import json
import os
from remedies import get_remedy

# Cache the model loading
@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path="crop_disease_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

@st.cache_resource
def load_labels():
    with open("labels.json") as f:
        return json.load(f)

# Load model and labels
try:
    interpreter = load_model()
    labels = load_labels()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading model: {str(e)}")

st.set_page_config(page_title="Smart Farming Assistant", layout="wide")

# Language selection
col1, col2 = st.columns([0.9, 0.1])
with col2:
    language = st.selectbox("Language", ["English", "Tamil"], key="lang")
lang_code = "en" if language == "English" else "ta"

# Title
st.title("🌾 AI Smart Farming Assistant" if lang_code == "en" else "🌾 AI ஸ்மார்ட் பண்ணை உதவியாளர்")

# Tabs
tab1, tab2, tab3 = st.tabs(
    ["Disease Detection", "Sensors", "Irrigation"] if lang_code == "en"
    else ["நோய் கண்டறிதல்", "சென்சர்கள்", "நீர்ப்பாசனம்"]
)

# TAB 1: Disease Detection
with tab1:
    st.subheader("Upload crop image for disease detection" if lang_code == "en" else "நோய் கண்டறிய பயிர் படத்தை பதிவேற்றவும்")
    
    uploaded_file = st.file_uploader("Choose image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file and model_loaded:
        image = Image.open(uploaded_file).resize((224, 224))
        st.image(image, caption="Uploaded Image" if lang_code == "en" else "பதிவேற்றப்பட்ட படம்")
        
        # Predict
        input_data = np.array(image, dtype=np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        predicted_class = labels[np.argmax(output[0])]
        confidence = np.max(output[0]) * 100
        
        # Status indicator
        if "healthy" in predicted_class.lower():
            status = "🟢 Healthy" if lang_code == "en" else "🟢 ஆரோக்கியம்"
        elif "Early" in predicted_class:
            status = "🟡 Early Stage Disease" if lang_code == "en" else "🟡 ஆரம்ப நோய்"
        else:
            status = "🔴 Advanced Disease" if lang_code == "en" else "🔴 முற்றிய நோய்"
        
        st.metric("Status", status)
        st.metric("Disease/Condition", predicted_class.replace("_", " "))
        st.metric("Confidence", f"{confidence:.1f}%")
        
        # Remedy
        remedy = get_remedy(predicted_class, lang_code)
        st.success(f"💡 Remedy: {remedy}")
    elif uploaded_file and not model_loaded:
        st.error("Model not loaded. Please check requirements.")

# TAB 2: Sensors
with tab2:
    st.subheader("Environmental Monitoring" if lang_code == "en" else "சுற்றுச்சூழல் கண்காணிப்பு")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        temp = st.slider("Temperature (°C)", 15, 40, 28)
        st.metric("🌡️ Temperature", f"{temp}°C")
    
    with col2:
        humidity = st.slider("Humidity (%)", 20, 95, 65)
        st.metric("💧 Humidity", f"{humidity}%")
    
    with col3:
        soil = st.slider("Soil Moisture", 0, 100, 45)
        moisture_status = "Wet 💧" if soil > 60 else ("Optimal 👍" if soil > 40 else "Dry 🌵")
        st.metric("🌱 Soil Moisture", f"{soil}% ({moisture_status})")

# TAB 3: Irrigation
with tab3:
    st.subheader("Irrigation Control" if lang_code == "en" else "நீர்ப்பாசனம் கட்டுப்பாடு")
    
    st.write("Automatic watering when soil gets dry:" if lang_code == "en" else "மண் வறண்டால் தானியங்கி நீர்ப்பாசனம்:")
    
    soil_dry_threshold = st.slider("Dry threshold (%)", 0, 100, 30)
    pump_status = st.checkbox("Enable Auto Irrigation" if lang_code == "en" else "தானியங்கி நீர்ப்பாசனத்தை இயக்கவும்")
    
    if pump_status:
        st.success("✅ Irrigation System Active" if lang_code == "en" else "✅ நீர்ப்பாசன முறை செயல்பாட்டில் உள்ளது")
    else:
        st.info("⏸️ Irrigation System Inactive" if lang_code == "en" else "⏸️ நீர்ப்பாசன முறை செயல்நிறுத்தப்பட்டுள்ளது")

st.markdown("---")
st.caption("Smart Farming Assistant v1.0 | Powered by AI" if lang_code == "en" else "ஸ்மார்ட் பண்ணை உதவியாளர் v1.0 | AI மூலம் இயங்கும்")
