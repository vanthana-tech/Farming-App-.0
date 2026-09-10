import gradio as gr
from PIL import Image
import numpy as np
import tensorflow as tf
import json
from gtts import gTTS
import io
from remedies import get_remedy
import threading
import time
from datetime import datetime
import os

interpreter = tf.lite.Interpreter(model_path="crop_disease_model.tflite")
interpreter.allocate_tensors()

with open("labels.json") as f:
    labels = json.load(f)

# Global sensor data with timestamps
sensor_data = {
    "temperature": 28.5,
    "humidity": 65,
    "soil_moisture": 45,
    "light": 850,
    "last_update": datetime.now().strftime("%H:%M:%S")
}

# Alert thresholds
ALERTS = {
    "temperature": {"min": 15, "max": 35, "unit": "°C"},
    "humidity": {"min": 30, "max": 90, "unit": "%"},
    "soil_moisture": {"min": 20, "max": 100, "unit": "%"},
    "light": {"min": 100, "max": 10000, "unit": "lux"}
}

notifications = []

def check_sensor_alerts():
    """Check sensors and generate alerts"""
    global notifications
    
    alerts = []
    
    # Temperature check
    if sensor_data["temperature"] < ALERTS["temperature"]["min"]:
        alerts.append(f"🔵 COLD ALERT: Temperature {sensor_data['temperature']}°C is too low!")
    elif sensor_data["temperature"] > ALERTS["temperature"]["max"]:
        alerts.append(f"🔴 HOT ALERT: Temperature {sensor_data['temperature']}°C is too high!")
    
    # Humidity check
    if sensor_data["humidity"] < ALERTS["humidity"]["min"]:
        alerts.append(f"🟠 DRY ALERT: Humidity {sensor_data['humidity']}% is too low!")
    elif sensor_data["humidity"] > ALERTS["humidity"]["max"]:
        alerts.append(f"🔵 WET ALERT: Humidity {sensor_data['humidity']}% is too high! Risk of fungal disease!")
    
    # Soil moisture check
    if sensor_data["soil_moisture"] < ALERTS["soil_moisture"]["min"]:
        alerts.append(f"🔴 DROUGHT ALERT: Soil moisture {sensor_data['soil_moisture']}% is critical! Water immediately!")
    elif sensor_data["soil_moisture"] > 80:
        alerts.append(f"🔵 WATERLOG ALERT: Soil is too wet! Reduce watering!")
    
    # Light check
    if sensor_data["light"] < ALERTS["light"]["min"]:
        alerts.append(f"🟠 LOW LIGHT: Insufficient light {sensor_data['light']} lux. Move plant to brighter location!")
    
    return alerts

def farmie_says(alert_message):
    """Farmie gives voice alert"""
    message = f"Attention farmer! {alert_message}. I'm Farmie, taking care of your crops! 💚"
    
    try:
        tts = gTTS(text=message, lang='en', slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return message, (16000, fp.read())
    except:
        return message, None

def get_live_notifications():
    """Get all live notifications"""
    alerts = check_sensor_alerts()
    
    notification_text = f"""
🔔 LIVE SENSOR ALERTS 🔔
Last Updated: {sensor_data['last_update']}

"""
    
    if not alerts:
        notification_text += "✅ All sensors are PERFECT! Your crops are healthy! 💚🌿"
    else:
        for alert in alerts:
            notification_text += f"• {alert}\n"
    
    notification_text += f"""

📊 Current Readings:
🌡️ Temperature: {sensor_data['temperature']}°C (Optimal: 15-35°C)
💧 Humidity: {sensor_data['humidity']}% (Optimal: 30-90%)
🌱 Soil Moisture: {sensor_data['soil_moisture']}% (Optimal: 20-100%)
☀️ Light: {sensor_data['light']} lux (Optimal: 100-10000)

💚 Farmie is monitoring your farm 24/7!
"""
    
    return notification_text

def simulate_sensor_update():
    """Simulate real sensor data updates"""
    import random
    
    global sensor_data
    
    # Simulate varying sensor readings
    sensor_data["temperature"] = round(25 + random.uniform(-5, 8), 1)
    sensor_data["humidity"] = round(60 + random.uniform(-20, 30), 1)
    sensor_data["soil_moisture"] = round(40 + random.uniform(-15, 25), 1)
    sensor_data["light"] = round(800 + random.uniform(-300, 400), 0)
    sensor_data["last_update"] = datetime.now().strftime("%H:%M:%S")
    
    return get_live_notifications()

def detect_disease(image, language):
    if image is None:
        return "📸 Please upload a crop image first!", None
    
    image_resized = Image.fromarray(image).resize((224, 224))
    input_data = np.array(image_resized, dtype=np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    predicted_class = labels[np.argmax(output[0])]
    confidence = np.max(output[0]) * 100
    remedy = get_remedy(predicted_class, "ta" if language == "Tamil" else "en")
    
    result = f"""
🌾 DISEASE DETECTED!
━━━━━━━━━━━━━━━━━━━━
🔴 Disease: {predicted_class}
📊 Confidence: {confidence:.1f}%

💡 Farmie's Treatment Plan:
{remedy}

🩺 Recommendation: Apply treatment immediately!
"""
    
    try:
        treatment = f"Attention! Your {predicted_class} plant needs treatment. {remedy}"
        tts = gTTS(text=treatment, lang='ta' if language == "Tamil" else 'en', slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return result, (16000, fp.read())
    except:
        return result, None

def talk_to_farmie(user_message, language):
    """Chat with Farmie"""
    
    responses = {
        "hi": "Hi farmer! 👋 I'm Farmie, your AI crop care assistant! I'm here to help! 💚",
        "hello": "Hello! 😊 I'm Farmie! What do you need help with?",
        "water": "💧 Your crops need water in early morning or evening! Current soil moisture is " + str(sensor_data['soil_moisture']) + "%",
        "fertilizer": "🌿 Use organic fertilizer every 2 weeks! Your plants will love it!",
        "disease": "🦠 Check your plants for yellow spots or wilting. Upload an image and I'll diagnose!",
        "alerts": "🔔 " + get_live_notifications(),
        "help": "📚 I can help with:\n• Disease detection\n• Watering advice\n• Sensor monitoring\n• Irrigation tips\nJust ask! 💚"
    }
    
    response = "That's interesting! 🤔 Ask me about disease, water, fertilizer, or alerts! 💚"
    for key, val in responses.items():
        if key in user_message.lower():
            response = val
            break
    
    try:
        tts = gTTS(text=response, lang='ta' if language == "Tamil" else 'en', slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        audio = (16000, fp.read())
    except:
        audio = None
    
    return response, audio

# Beautiful Custom CSS
custom_css = """
body {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
}

.gradio-container {
    background: white !important;
    border-radius: 20px !important;
    box-shadow: 0 10px 40px rgba(0,0,0,0.3) !important;
    max-width: 1200px !important;
}

.gr-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: bold !important;
    transition: all 0.3s !important;
}

.gr-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4) !important;
}

.gr-textbox, .gr-image {
    border-radius: 12px !important;
    border: 2px solid #ddd !important;
}

.alert-box {
    background: linear-gradient(135deg, #fff5e6 0%, #ffe6cc 100%) !important;
    border-left: 5px solid #ff6b6b !important;
    padding: 15px !important;
    border-radius: 10px !important;
}

h1, h2, h3 {
    color: #333 !important;
}
"""

# Build the app
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 🌾 AI Smart Farming Assistant
    ### Meet **Farmie** 👧💚 - Your AI Girl Farming Helper
    #### Real-time Sensor Monitoring + Live Alerts
    """)
    
    language = gr.Dropdown(["English", "Tamil"], value="English", label="🌏 Language")
    
    with gr.Tabs():
        # Live Alerts
        with gr.Tab("🔔 LIVE ALERTS"):
            gr.Markdown("### Farmie's Real-time Sensor Monitoring 💚")
            
            notifications_display = gr.Textbox(
                get_live_notifications(),
                label="📡 Live Sensor Alerts",
                lines=12,
                interactive=False
            )
            
            with gr.Row():
                refresh_alerts = gr.Button("🔄 Refresh Alerts NOW", variant="primary", size="lg")
                auto_refresh = gr.Checkbox(label="🔁 Auto Refresh (Every 5s)", value=False)
            
            refresh_alerts.click(simulate_sensor_update, outputs=notifications_display)
        
        # Disease Detection
        with gr.Tab("🔍 Disease Detection"):
            gr.Markdown("### Upload crop image and Farmie will diagnose 🌱")
            
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(sources=["webcam", "upload"], type="numpy", label="📸 Crop")
                with gr.Column():
                    disease_output = gr.Textbox(label="💚 Farmie's Diagnosis", lines=8, interactive=False)
                    disease_audio = gr.Audio(label="🎤 Farmie Speaking")
            
            detect_btn = gr.Button("🔎 Ask Farmie to Detect", size="lg", variant="primary")
            detect_btn.click(detect_disease, [image_input, language], [disease_output, disease_audio])
        
        # Chat with Farmie
        with gr.Tab("💬 Chat with Farmie"):
            gr.Markdown("### Talk to Farmie anytime! 💚🌻")
            
            with gr.Row():
                with gr.Column():
                    user_input = gr.Textbox(
                        placeholder="Ask about disease, water, fertilizer, alerts...",
                        label="💬 Your Question",
                        lines=3
                    )
                    chat_btn = gr.Button("💬 Talk to Farmie", size="lg", variant="primary")
                
                with gr.Column():
                    farmie_response = gr.Textbox(label="💚 Farmie's Reply", lines=5, interactive=False)
                    farmie_voice = gr.Audio(label="🎤 Farmie Speaking", type="numpy")
            
            chat_btn.click(talk_to_farmie, [user_input, language], [farmie_response, farmie_voice])
        
        # Sensor Details
        with gr.Tab("📊 Sensor Details"):
            gr.Markdown("### Detailed Sensor Information 📡")
            
            detail_text = f"""
🌡️ TEMPERATURE: {sensor_data['temperature']}°C
   Status: {'✅ Normal' if 15 <= sensor_data['temperature'] <= 35 else '⚠️ Alert'}
   Optimal Range: 15-35°C
   Role: Affects growth rate & flowering

💧 HUMIDITY: {sensor_data['humidity']}%
   Status: {'✅ Normal' if 30 <= sensor_data['humidity'] <= 90 else '⚠️ Alert'}
   Optimal Range: 30-90%
   Role: Prevents fungal diseases

🌱 SOIL MOISTURE: {sensor_data['soil_moisture']}%
   Status: {'✅ Normal' if 20 <= sensor_data['soil_moisture'] <= 80 else '⚠️ Alert'}
   Optimal Range: 20-80%
   Role: Root hydration & nutrient uptake

☀️ LIGHT: {sensor_data['light']} lux
   Status: {'✅ Normal' if 100 <= sensor_data['light'] <= 10000 else '⚠️ Alert'}
   Optimal Range: 100-10000 lux
   Role: Photosynthesis & growth
"""
            
            gr.Textbox(detail_text, lines=15, interactive=False, label="📊 Sensor Details")
            refresh_detail = gr.Button("🔄 Refresh Details")
            refresh_detail.click(simulate_sensor_update)

app.launch(server_name="0.0.0.0", server_port=7860, share=False)
