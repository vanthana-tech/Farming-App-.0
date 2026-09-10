import gradio as gr
from PIL import Image
import numpy as np
import tensorflow as tf
import json
from gtts import gTTS
import io
from remedies import get_remedy

# Load model
interpreter = tf.lite.Interpreter(model_path="crop_disease_model.tflite")
interpreter.allocate_tensors()

with open("labels.json") as f:
    labels = json.load(f)

def disease_detection(image, language):
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
    
    result = f"🌾 Disease: {predicted_class}\n\nConfidence: {confidence:.1f}%\n\nRemedy: {remedy}"
    
    try:
        tts = gTTS(text=remedy, lang='ta' if language == "Tamil" else 'en')
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return result, (16000, audio_fp.read())
    except:
        return result, None

def farming_assistant(question):
    q = question.lower()
    advice = {
        "water": "Water crops daily in early morning or evening.",
        "fertilizer": "Use organic fertilizer every 2 weeks.",
        "pest": "Use neem oil spray for pest control.",
        "soil": "Test soil every 3 months. Keep pH 6-7.",
        "disease": "Remove affected leaves immediately."
    }
    
    for key, val in advice.items():
        if key in q:
            return f"Assistant: {val}"
    return "Ask about: water, fertilizer, pest, soil, or disease"

with gr.Blocks() as demo:
    gr.Markdown("# 🌾 Smart Farming System")
    
    with gr.Tabs():
        with gr.TabItem("Disease Detection"):
            image = gr.Image(sources=["camera", "upload"], type="numpy")
            lang = gr.Dropdown(["English", "Tamil"])
            btn = gr.Button("Detect")
            output_text = gr.Textbox()
            output_audio = gr.Audio()
            btn.click(disease_detection, [image, lang], [output_text, output_audio])
        
        with gr.TabItem("Farming Assistant"):
            question = gr.Textbox(placeholder="Ask about farming")
            answer = gr.Textbox()
            ask_btn = gr.Button("Ask")
            ask_btn.click(farming_assistant, question, answer)

demo.launch(server_name="0.0.0.0", server_port=7860)


