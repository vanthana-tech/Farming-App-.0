"""
AI-Powered Smart Farming Assistant - Demo Dashboard
Run with: streamlit run app.py
Requires (in the same folder):
- crop_disease_model.tflite   (your trained model, downloaded from Colab)
- labels.json                 (your class labels, downloaded from Colab)
- remedies.py                 (included alongside this file)
"""
import json
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from remedies import get_remedy, evaluate_sensors, STATUS_COLORS

st.set_page_config(page_title="Smart Farming Assistant", page_icon="🌱", layout="centered")

# ---------- Load model + labels (cached so it only loads once) ----------
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="crop_disease_model.tflite")
    interpreter.allocate_tensors()
    with open("labels.json") as f:
        labels = json.load(f)
    # JSON keys are strings; convert back to int-indexed dict
    labels = {int(k): v for k, v in labels.items()}
    return interpreter, labels


def predict(interpreter, labels, pil_image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    size = input_details[0]["shape"][1:3]
    img = pil_image.resize((size[1], size[0]))
    arr = np.array(img).astype("float32")
    arr = (arr / 127.5) - 1.0  # match MobileNetV2 preprocessing
    arr = np.expand_dims(arr, axis=0)
    interpreter.set_tensor(input_details[0]["index"], arr)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])[0]
    predicted_idx = int(np.argmax(output))
    confidence = float(np.max(output)) * 100
    return labels[predicted_idx], confidence


# ---------- Sidebar: language + simulated sensors ----------
st.sidebar.header("Settings")
language = st.sidebar.radio("Language / மொழி", ["English", "தமிழ்"])
lang_code = "en" if language == "English" else "ta"

st.sidebar.markdown("---")
st.sidebar.subheader("Sensor readings (live from field unit)")
# In the real device these come from the ESP32 over serial/WiFi.
# For this demo, sliders simulate live sensor input.
soil_moisture = st.sidebar.slider("Soil moisture (%)", 0, 100, 45)
temperature = st.sidebar.slider("Temperature (°C)", 15, 50, 29)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 65)

# ---------- Main title ----------
st.title("🌱 Smart Farming Assistant" if lang_code == "en" else "🌱 ஸ்மார்ட் விவசாய உதவியாளர்")
st.caption(
    "Upload or capture a leaf photo to check your crop's health"
    if lang_code == "en"
    else "உங்கள் பயிரின் ஆரோக்கியத்தை சரிபார்க்க இலை புகைப்படத்தை பதிவேற்றவும்"
)

# ---------- Image input ----------
img_file = st.camera_input("Scan a leaf" if lang_code == "en" else "இலையை ஸ்கேன் செய்யவும்")
if img_file is None:
    img_file = st.file_uploader(
        "Or upload a photo" if lang_code == "en" else "அல்லது புகைப்படத்தை பதிவேற்றவும்",
        type=["jpg", "jpeg", "png"],
    )

st.markdown("---")

if img_file is not None:
    image = Image.open(img_file).convert("RGB")
    st.image(image, caption="Scanned leaf" if lang_code == "en" else "ஸ்கேன் செய்யப்பட்ட இலை", width=300)

    try:
        interpreter, labels = load_model()
        predicted_class, confidence = predict(interpreter, labels, image)
    except FileNotFoundError:
        st.error(
            "Model files not found. Place crop_disease_model.tflite and "
            "labels.json in this folder."
        )
        st.stop()

    st.subheader("Disease / Pest Check" if lang_code == "en" else "நோய் / பூச்சி பரிசோதனை")
    if confidence < 70:
        note = (
            "Possible issue detected - please verify visually or rescan"
            if lang_code == "en"
            else "சாத்தியமான பிரச்சனை கண்டறியப்பட்டது - தயவுசெய்து மீண்டும் சரிபார்க்கவும்"
        )
        st.warning(f"⚠️ {note} (confidence: {confidence:.1f}%)")
    else:
        result = get_remedy(predicted_class, lang_code)
        emoji = STATUS_COLORS.get(result["status"], "ℹ️")
        st.markdown(f"### {emoji}{result['message']}")
        st.write(f"**{'Confidence' if lang_code == 'en' else 'நம்பகத்தன்மை'}:** {confidence:.1f}%")
        if result["remedy"]:
            st.info(result["remedy"])

    # ---------- Sensor-based condition ----------
    st.markdown("---")
    st.subheader("Field Conditions" if lang_code == "en" else "வயல் நிலைமைகள்")
    col1, col2, col3 = st.columns(3)
    col1.metric("Soil Moisture", f"{soil_moisture}%")
    col2.metric("Temperature", f"{temperature}°C")
    col3.metric("Humidity", f"{humidity}%")

    sensor_condition = evaluate_sensors(soil_moisture, temperature)
    sensor_result = get_remedy(sensor_condition, lang_code)
    emoji = STATUS_COLORS.get(sensor_result["status"], "ℹ️")
    st.markdown(f"### {emoji}{sensor_result['message']}")
    if sensor_result["remedy"]:
        st.info(sensor_result["remedy"])
else:
    st.info(
        "Take a photo or upload one to get started."
        if lang_code == "en"
        else "தொடங்க ஒரு புகைப்படத்தை எடுக்கவும் அல்லது பதிவேற்றவும்."
    )
