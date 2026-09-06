import gradio as gr
from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite
import json
from gtts import gTTS
import tempfile
from remedies import get_remedy


# -----------------------------
# Load TFLite model
# -----------------------------
interpreter = tflite.Interpreter(model_path="crop_disease_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]["shape"]
input_dtype = input_details[0]["dtype"]

IMG_HEIGHT = int(input_shape[1])
IMG_WIDTH = int(input_shape[2])


# -----------------------------
# Load labels
# -----------------------------
with open("labels.json", "r") as f:
    labels = json.load(f)


def get_label(index):
    """Works with either a list or dictionary labels.json."""
    if isinstance(labels, list):
        return labels[index]

    if isinstance(labels, dict):
        if str(index) in labels:
            return labels[str(index)]

        if index in labels:
            return labels[index]

    return f"Class {index}"


# -----------------------------
# Disease prediction
# -----------------------------
def predict_disease(image, language):

    if image is None:
        return "📷 Please upload or capture a crop image.", None

    try:
        # Convert image to RGB
        image = Image.fromarray(image).convert("RGB")

        # Resize according to model input
        image = image.resize((IMG_WIDTH, IMG_HEIGHT))

        # Prepare input
        if input_dtype == np.uint8:
            input_data = np.array(image, dtype=np.uint8)
        else:
            input_data = np.array(image, dtype=np.float32) / 255.0

        input_data = np.expand_dims(input_data, axis=0)

        # Run model
        interpreter.set_tensor(
            input_details[0]["index"],
            input_data
        )

        interpreter.invoke()

        output = interpreter.get_tensor(
            output_details[0]["index"]
        )

        # Prediction
        predicted_index = int(np.argmax(output[0]))
        confidence = float(np.max(output[0])) * 100

        predicted_class = get_label(predicted_index)

        # Confidence protection
        if confidence < 50:
            result = (
                "⚠️ Unable to confidently identify the disease.\n\n"
                f"📊 Confidence: {confidence:.1f}%\n\n"
                "📷 Please upload a clearer photo of the affected leaf."
            )
            return result, None

        # Language
        lang_code = "ta" if language == "Tamil" else "en"

        # Remedy
        remedy = get_remedy(predicted_class, lang_code)

        result = (
            f"🌾 Disease: {predicted_class}\n\n"
            f"📊 Confidence: {confidence:.1f}%\n\n"
            f"💡 Remedy:\n{remedy}"
        )

        # AI Voice
        try:
            tts = gTTS(
                text=remedy,
                lang=lang_code,
                slow=False
            )

            audio_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp3"
            )

            tts.save(audio_file.name)

            return result, audio_file.name

        except Exception:
            return result, None

    except Exception as e:
        return (
            f"❌ Error while detecting the disease:\n{str(e)}",
            None
        )


# -----------------------------
# Gradio interface
# -----------------------------
interface = gr.Interface(
    fn=predict_disease,

    inputs=[
        gr.Image(
            sources=["camera", "upload"],
            type="numpy",
            label="📷 Upload or Capture Crop Image"
        ),

        gr.Dropdown(
            choices=["English", "Tamil"],
            value="English",
            label="🌐 Language"
        )
    ],

    outputs=[
        gr.Markdown(
            label="🌾 Detection Result"
        ),

        gr.Audio(
            label="🔊 AI Voice",
            type="filepath"
        )
    ],

    title="🌾 Smart Farming Disease Detection",

    description=(
        "📷 Upload or capture a crop image to detect "
        "plant diseases using AI and receive remedies "
        "in English or Tamil."
    )
)


# -----------------------------
# Launch
# -----------------------------
import os

interface.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)



