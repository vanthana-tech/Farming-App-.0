
import gradio as gr
from PIL import Image
import numpy as np
import tensorflow as tf
from ai_edge_litert.interpreter
import Interpreter
import json
from gtts import gTTS
import io
from remedies import get_remedy

# Load model
interpreter = Interpreter(model_path="crop_disease_model.tflite")
interpreter.allocate_tensors()

with open("labels.json") as f:
    labels = json.load(f)

# AI Assistant Knowledge Base
FARMING_ADVICE = {
    "water": "Water your crops daily in early morning or evening. Avoid midday watering.",
    "fertilizer": "Use organic fertilizer every 2 weeks. NPK ratio 20:20:20 is good for most crops.",
    "pest": "Use neem oil spray for pest control. Spray in evening for best results.",
    "soil": "Test soil every 3 months. Keep pH between 6-7 for most vegetables.",
    "temperature": "Tomatoes grow best at 21-24°C. Potatoes at 15-20°C.",
    "humidity": "Ideal humidity is 60-70%. Use mulch to retain moisture.",
    "disease": "Remove affected leaves immediately. Improve air circulation to prevent diseases.",
    "harvest": "Harvest tomatoes when fully ripe. Potatoes when leaves turn yellow."
}

def disease_detection(image, language):
    """Detect crop disease from image"""
    image = Image.fromarray(image).resize((224, 224))
    input_data = np.array(image, dtype=np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    predicted_class = labels[np.argmax(output[0])]
    confidence = np.max(output[0]) * 100
    lang_code = "ta" if language == "Tamil" else "en"
    remedy = get_remedy(predicted_class, lang_code)
    
    result = f"🌾 **Disease: {predicted_class}**\n\n📊 Confidence: {confidence:.1f}%\n\n💡 **Remedy:**\n{remedy}"
    
    try:
        tts = gTTS(text=remedy, lang='ta' if language == "Tamil" else 'en', slow=False)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return result, (16000, audio_fp.read())
    except:
        return result, None

def farming_assistant(question):
    """AI Assistant for farming advice"""
    question = question.lower()
    
    for keyword, answer in FARMING_ADVICE.items():
        if keyword in question:
            return f"🤖 **Farming Assistant:**\n\n{answer}"
    
    return "🤖 **Farming Assistant:**\n\nI don't have info on that. Ask about: water, fertilizer, pest, soil, temperature, humidity, disease, or harvest."

# Gradio Interface
with gr.Blocks(title="🌾 Smart Farming System") as demo:
    gr.Markdown("# 🌾 AI Smart Farming Assistant")
    
    with gr.Tabs():
        # Tab 1: Disease Detection
        with gr.TabItem("🔍 Disease Detection"):
            with gr.Row():
                image_input = gr.Image(sources=["camera", "upload"], type="numpy", label="Crop Image")
                language = gr.Dropdown(["English", "Tamil"], label="Language")
            
            detect_btn = gr.Button("Detect Disease", variant="primary")
            
            with gr.Row():
                text_output = gr.Textbox(label="Result", lines=6)
                audio_output = gr.Audio(label="AI Voice")
            
            detect_btn.click(disease_detection, inputs=[image_input, language], outputs=[text_output, audio_output])
        
        # Tab 2: Farming Assistant
        with gr.TabItem("🤖 Farming Assistant"):
            gr.Markdown("Ask questions about: water, fertilizer, pest, soil, temperature, humidity, disease, harvest")
            
            question_input = gr.Textbox(label="Ask a farming question", placeholder="e.g., How to water crops?")
            assistant_output = gr.Textbox(label="Assistant Answer", lines=4)
            ask_btn = gr.Button("Ask", variant="primary")
            
            ask_btn.click(farming_assistant, inputs=question_input, outputs=assistant_output)

demo.launch()
