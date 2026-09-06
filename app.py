import gradio as gr
from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite
import json
from gtts import gTTS
import io
from remedies import get_remedy

# Load model
interpreter = tflite.Interpreter(model_path="crop_disease_model.tflite")
interpreter.allocate_tensors()

with open("labels.json") as f:
    labels = json.load(f)

def predict_disease(image, language):
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
    
    # Text result
    result = f"🌾 **Disease: {predicted_class}**\n\n📊 Confidence: {confidence:.1f}%\n\n💡 **Remedy:**\n{remedy}"
    
    # AI Voice (Google Text-to-Speech)
    try:
        tts = gTTS(text=remedy, lang='ta' if language == "Tamil" else 'en', slow=False)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        audio_bytes = audio_fp.read()
        return result, (16000, audio_bytes)
    except:
        return result, None

# Gradio interface
interface = gr.Interface(
    fn=predict_disease,
    inputs=[
        gr.Image(sources=["camera", "upload"], type="numpy"),
        gr.Dropdown(["English", "Tamil"])
    ],
    outputs=[
        gr.Textbox(label="Result"),
        gr.Audio(label="AI Voice")
    ],
    title="🌾 Smart Farming Disease Detection",
    description="Use camera to detect crop diseases + AI voice remedy"
)

interface.launch()
