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

def detect_disease(image, language):
    """AI Girl Farmie detects disease and speaks"""
    
    if image is None:
        return "❌ Please upload an image first! 📸", None
    
    try:
        # Process image
        img = Image.fromarray(image).resize((224, 224))
        input_data = np.array(img, dtype=np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)
        
        # AI Detection
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        # Get result
        predicted_class = labels[np.argmax(output[0])]
        confidence = np.max(output[0]) * 100
        lang_code = "ta" if language == "Tamil" else "en"
        remedy = get_remedy(predicted_class, lang_code)
        
        # Farmie's response
        if "healthy" in predicted_class.lower():
            if language == "Tamil":
                title = "✅ வாழ்த்துக்கள்! உங்கள் பயிர் ஆரோக்கியமாக உள்ளது! 🌿"
                result = f"📊 நிலை: HEALTHY\n🌾 உங்கள் பயிர் அற்புதமாக வளர்ந்து வருகிறது!"
                farmie_voice = "வாழ்த்துக்கள்! உங்கள் பயிர் ஆரோக்கியமாக உள்ளது!"
            else:
                title = "✅ Congratulations! Your Crop is Healthy! 🌿"
                result = f"📊 Status: HEALTHY\n🌾 Your crop is growing beautifully!"
                farmie_voice = "Congratulations! Your crop is healthy and strong!"
        else:
            if language == "Tamil":
                title = f"⚠️ நோய் கண்டறியப்பட்டுவிட்டது! {predicted_class}"
                result = f"🔴 நோய்: {predicted_class}\n📊 நம்பிக்கை: {confidence:.1f}%\n\n💡 சிகிச்சை திட்டம்:\n{remedy}"
                farmie_voice = f"கவனம்! உங்கள் பயிரில் {predicted_class} நோய் உள்ளது. உடனே சிகிச்சை தேவை!"
            else:
                title = f"⚠️ Disease Detected! {predicted_class}"
                result = f"🔴 Disease: {predicted_class}\n📊 Confidence: {confidence:.1f}%\n\n💡 Treatment:\n{remedy}"
                farmie_voice = f"Attention! Your crop has {predicted_class}. Immediate treatment needed!"
        
        # Generate voice
        audio = None
        try:
            tts = gTTS(text=farmie_voice, lang='ta' if language == "Tamil" else 'en', slow=False)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            return title + "\n\n" + result, (16000, fp.read())
        except:
            return title + "\n\n" + result, None
    
    except Exception as e:
        return f"❌ Error: {str(e)}", None

# Custom theme
custom_css = """
.gr-container { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
.gradio-container { 
    max-width: 900px !important; 
    background: white !important;
    border-radius: 20px !important;
}
.gr-button { 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border-radius: 15px !important;
}
.gr-button:hover { transform: translateY(-3px); }
.gr-image { border-radius: 15px !important; }
.gr-textbox { border-radius: 15px !important; }
"""

# Build beautiful app
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="🌾 Farmie - AI Crop Care") as app:
    
    gr.Markdown("""
    # 👧 Farmie - AI Crop Care Assistant 💚
    ### Your AI Girl Who Cares About Your Crops 🌾
    """)
    
    language = gr.Dropdown(
        ["English", "Tamil"],
        value="English",
        label="🌏 Select Language / மொழியைத் தேர்ந்தெடுக்கவும்"
    )
    
    gr.Markdown("### 📸 Upload Your Crop Image / உங்கள் பயிரின் படத்தை பதிவேற்றவும்")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                sources=["webcam", "upload"],
                type="numpy",
                label="Crop Image / பயிரின் படம்"
            )
        
        with gr.Column():
            analysis_output = gr.Textbox(
                label="🤖 Farmie's Analysis / ஃபார்மியின் பகுப்பாய்வு",
                lines=8,
                interactive=False
            )
            voice_output = gr.Audio(
                label="🎤 Farmie Speaking / ஃபார்மி பேசுவது",
                type="numpy"
            )
    
    analyze_btn = gr.Button(
        "🔎 Ask Farmie to Analyze / ஃபார்மியிடம் பகுப்பாய்வு கேளுங்கள்",
        variant="primary",
        size="lg"
    )
    
    analyze_btn.click(
        detect_disease,
        inputs=[image_input, language],
        outputs=[analysis_output, voice_output]
    )
    
    gr.Markdown("""
    ---
    ### 💚 About Farmie
    Farmie is your AI crop care assistant girl. She analyzes your crop images 
    and gives treatment recommendations with voice guidance in Tamil and English.
    
    ### 💚 ஃபார்மி பற்றி
    ஃபார்மி உங்கள் AI பயிர் பராமரிப்பு உதவியாள். 
    அவர் உங்கள் பயிரின் படங்களை பகுப்பாய்வு செய்து சிகிச்சை பரிந்துரைகளை வழங்குகிறார்.
    """)

app.launch(server_name="0.0.0.0", server_port=7860)
