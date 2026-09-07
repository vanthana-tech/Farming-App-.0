import gradio as gr
from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite
import json
from remedies import get_remedy
import os

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
    remedy = get_remedy(predicted_class, "ta" if language == "Tamil" else "en")
    
    return f"**{predicted_class}** ({confidence:.1f}%)\n\n💡 {remedy}"

interface = gr.Interface(
    fn=predict_disease,
    inputs=[gr.Image(), gr.Dropdown(["English", "Tamil"])],
    outputs="text",
    title="🌾 Smart Farming Disease Detection",
    description="Upload a crop image to detect diseases"
)

interface.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)
