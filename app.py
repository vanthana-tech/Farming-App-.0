from flask import Flask, render_template_string, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import json
from gtts import gTTS
import io
import base64
from remedies import get_remedy

app = Flask(__name__)

interpreter = tf.lite.Interpreter(model_path="crop_disease_model.tflite")
interpreter.allocate_tensors()

with open("labels.json") as f:
    labels = json.load(f)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Farmie - AI Crop Care</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 25px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        
        header {
            text-align: center;
            margin-bottom: 40px;
        }
        
        .avatar {
            font-size: 80px;
            margin-bottom: 20px;
        }
        
        h1 {
            color: #333;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .subtitle {
            color: #666;
            font-size: 1.1em;
        }
        
        .language-btn {
            position: absolute;
            top: 20px;
            right: 20px;
            padding: 10px 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 20px;
            cursor: pointer;
            font-weight: bold;
        }
        
        .upload-box {
            border: 3px dashed #667eea;
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
            background: #f8f9ff;
        }
        
        .upload-box h2 {
            color: #333;
            margin-bottom: 10px;
        }
        
        .upload-box p {
            color: #666;
            margin-bottom: 20px;
        }
        
        #imageInput { display: none; }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 40px;
            border: none;
            border-radius: 20px;
            font-size: 1.1em;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.3);
        }
        
        .btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.5);
        }
        
        #imagePreview {
            display: none;
            max-width: 400px;
            border-radius: 15px;
            margin: 20px auto;
        }
        
        .detect-btn {
            width: 100%;
            margin-top: 20px;
            padding: 18px;
            font-size: 1.2em;
            display: none;
        }
        
        .result {
            background: #f8f9ff;
            border-radius: 20px;
            padding: 30px;
            margin-top: 30px;
            border-left: 5px solid #667eea;
            display: none;
        }
        
        .result.show {
            display: block;
            animation: slideIn 0.5s ease;
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .result-title {
            color: #667eea;
            font-size: 1.3em;
            margin-bottom: 15px;
            font-weight: bold;
        }
        
        .result-text {
            color: #333;
            line-height: 1.8;
            font-size: 1.05em;
            white-space: pre-wrap;
        }
        
        .farmie-msg {
            background: white;
            border: 2px solid #667eea;
            border-radius: 15px;
            padding: 20px;
            margin-top: 20px;
            color: #333;
            font-size: 1.05em;
        }
        
        audio {
            width: 100%;
            margin-top: 20px;
            border-radius: 10px;
        }
        
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        
        .spinner {
            border: 4px solid #f0f0f0;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <button class="language-btn" onclick="toggleLanguage()">🇬🇧 English / 🇮🇳 Tamil</button>
    
    <div class="container">
        <header>
            <div class="avatar">👧</div>
            <h1>Farmie 💚</h1>
            <p class="subtitle">Your AI Crop Care Assistant</p>
        </header>
        
        <div class="upload-box">
            <h2 id="uploadTitle">📸 Upload Your Crop Image</h2>
            <p id="uploadText">Choose from camera or gallery</p>
            <input type="file" id="imageInput" accept="image/*">
            <button class="btn" onclick="document.getElementById('imageInput').click()" id="uploadBtn">
                📤 Choose Image
            </button>
        </div>
        
        <img id="imagePreview" alt="Preview">
        <button class="btn detect-btn" onclick="detectDisease()" id="detectBtn">🔎 Ask Farmie to Analyze</button>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p id="loadingText" style="color: #667eea; margin-top: 15px; font-weight: bold;">Analyzing...</p>
        </div>
        
        <div class="result" id="result">
            <div class="result-title" id="resultTitle"></div>
            <div class="result-text" id="resultText"></div>
            <div class="farmie-msg" id="farmieMsg"></div>
            <audio id="audio" controls></audio>
        </div>
    </div>
    
    <script>
        let lang = 'english';
        
        function toggleLanguage() {
            lang = lang === 'english' ? 'tamil' : 'english';
            updateText();
        }
        
        function updateText() {
            const uploadBtn = document.getElementById('uploadBtn');
            const uploadTitle = document.getElementById('uploadTitle');
            const uploadText = document.getElementById('uploadText');
            const detectBtn = document.getElementById('detectBtn');
            
            if (lang === 'tamil') {
                uploadTitle.textContent = '📸 உங்கள் பயிரின் படத்தை பதிவேற்றவும்';
                uploadText.textContent = 'கேமராவிலிருந்து அல்லது கேலரியிலிருந்து தேர்ந்தெடுக்கவும்';
                uploadBtn.textContent = '📤 படத்தைத் தேர்ந்தெடுக்கவும்';
                detectBtn.textContent = '🔎 ஃபார்மியிடம் பகுப்பாய்வு கேளுங்கள்';
            } else {
                uploadTitle.textContent = '📸 Upload Your Crop Image';
                uploadText.textContent = 'Choose from camera or gallery';
                uploadBtn.textContent = '📤 Choose Image';
                detectBtn.textContent = '🔎 Ask Farmie to Analyze';
            }
        }
        
        document.getElementById('imageInput').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(event) {
                    document.getElementById('imagePreview').src = event.target.result;
                    document.getElementById('imagePreview').style.display = 'block';
                    document.getElementById('detectBtn').style.display = 'block';
                };
                reader.readAsDataURL(file);
            }
        });
        
        async function detectDisease() {
            const file = document.getElementById('imageInput').files[0];
            if (!file) {
                alert('Please select image');
                return;
            }
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').classList.remove('show');
            
            const formData = new FormData();
            formData.append('file', file);
            formData.append('language', lang);
            
            try {
                const response = await fetch('/api/detect', { method: 'POST', body: formData });
                const data = await response.json();
                
                document.getElementById('resultTitle').textContent = data.title;
                document.getElementById('resultText').textContent = data.result;
                document.getElementById('farmieMsg').textContent = data.farmie_msg;
                
                if (data.audio) {
                    document.getElementById('audio').src = 'data:audio/mp3;base64,' + data.audio;
                }
                
                document.getElementById('result').classList.add('show');
            } catch (error) {
                alert('Error: ' + error);
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/api/detect', methods=['POST'])
def detect():
    file = request.files['file']
    language = request.form.get('language', 'english')
    
    image = Image.open(file).resize((224, 224))
    input_data = np.array(image, dtype=np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    predicted_class = labels[np.argmax(output[0])]
    confidence = np.max(output[0]) * 100
    remedy = get_remedy(predicted_class, "ta" if language == "tamil" else "en")
    
if "healthy" in predicted_class.lower():
        if language == "tamil":
            title = "✅ வாழ்த்துக்கள்! ஆரோக்கியமாக உள்ளது!"
            result = (
                f"🌿 நிலை: ஆரோக்கியமான பயிர்\n"
                f"📊 நம்பிக்கை: {confidence:.1f}%\n\n"
                f"உங்கள் பயிர் ஆரோக்கியமாக உள்ளது. "
                f"தொடர்ந்து நல்ல பராமரிப்பை செய்யுங்கள்."
            )
            farmie_msg = (
                "🌱 ஃபார்மி: உங்கள் பயிர் ஆரோக்கியமாக இருக்கிறது! "
                "தொடர்ந்து தண்ணீர், ஊட்டச்சத்து மற்றும் பூச்சி கண்காணிப்பை செய்யுங்கள்."
            )
        else:
            title = "✅ Great! Your Crop is Healthy!"
            result = (
                f"🌿 Status: HEALTHY\n"
                f"📊 Confidence: {confidence:.1f}%\n\n"
                f"Your crop appears to be healthy. "
                f"Continue with good crop care and regular monitoring."
            )
            farmie_msg = (
                "🌱 Farmie: Your crop looks healthy! "
                "Keep monitoring it regularly and maintain proper watering and nutrition."
            )
        
else:
      if language == "tamil":
            title = f"⚠️ பயிர் நோய் கண்டறியப்பட்டது: {predicted_class}"
            result = (
                f"🌿 நோய்: {predicted_class}\n"
                f"📊 நம்பிக்கை: {confidence:.1f}%\n\n"
                f"💊 பரிந்துரைக்கப்பட்ட தீர்வு:\n{remedy}"
            )
            farmie_msg = (
                f"🌱 ஃபார்மி: உங்கள் பயிரில் {predicted_class} "
                f"கண்டறியப்பட்டுள்ளது. பரிந்துரைக்கப்பட்ட பராமரிப்பு முறைகளைப் பின்பற்றுங்கள்."
            )
        else:
            title = f"⚠️ Crop Disease Detected: {predicted_class}"
            result = (
                f"🌿 Disease: {predicted_class}\n"
                f"📊 Confidence: {confidence:.1f}%\n\n"
                f"💊 Recommended Remedy:\n{remedy}"
            )
            farmie_msg = (
                f"🌱 Farmie: {predicted_class} has been detected in your crop. "
                f"Please follow the recommended treatment and care steps."
            )

    # Generate voice output
    try:
        voice_text = farmie_msg

        if language == "tamil":
            tts = gTTS(text=voice_text, lang="ta")
        else:
            tts = gTTS(text=voice_text, lang="en")

        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)

        audio_base64 = base64.b64encode(
            audio_buffer.read()
        ).decode("utf-8")

    except Exception as e:
        print("TTS Error:", e)
        audio_base64 = None

    return jsonify({
        "title": title,
        "result": result,
        "farmie_msg": farmie_msg,
        "audio": audio_base64
    })

if "healthy" in predicted_class.lower():
    if language == "tamil":
        title = "✅ வாழ்த்துக்கள்! ஆரோக்கியமாக உள்ளது!"
        result = f"🌿 நிலை: ஆரோக்கியமான பயிர்\n📊 நம்பிக்கை: {confidence:.1f}%\n\nஉங்கள் பயிர் ஆரோக்கியமாக உள்ளது."
        farmie_msg = "🌱 ஃபார்மி: உங்கள் பயிர் ஆரோக்கியமாக இருக்கிறது! தொடர்ந்து நல்ல பராமரிப்பை செய்யுங்கள்."
    else:
        title = "✅ Great! Your Crop is Healthy!"
        result = f"🌿 Status: HEALTHY\n📊 Confidence: {confidence:.1f}%\n\nYour crop appears to be healthy."
        farmie_msg = "🌱 Farmie: Your crop looks healthy! Keep monitoring it regularly."

else:
    if language == "tamil":
        title = f"⚠️ பயிர் நோய் கண்டறியப்பட்டது: {predicted_class}"
        result = f"🌿 நோய்: {predicted_class}\n📊 நம்பிக்கை: {confidence:.1f}%\n\n💊 பரிந்துரைக்கப்பட்ட தீர்வு:\n{remedy}"
        farmie_msg = f"🌱 ஃபார்மி: உங்கள் பயிரில் {predicted_class} கண்டறியப்பட்டுள்ளது."
    else:
        title = f"⚠️ Crop Disease Detected: {predicted_class}"
        result = f"🌿 Disease: {predicted_class}\n📊 Confidence: {confidence:.1f}%\n\n💊 Recommended Remedy:\n{remedy}"
        farmie_msg = f"🌱 Farmie: {predicted_class} has been detected in your crop."
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )

