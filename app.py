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

# Load model
interpreter = tf.lite.Interpreter(model_path="crop_disease_model.tflite")
interpreter.allocate_tensors()

with open("labels.json") as f:
    labels = json.load(f)

# Sensor data storage
sensor_data = {
    "temperature": 28.5,
    "humidity": 65,
    "soil_moisture": 45,
    "light": 850,
    "pump_status": "OFF"
}

# HTML + CSS Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🌾 Smart Farming Assistant</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }
        
        header {
            text-align: center;
            margin-bottom: 40px;
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
        
        .language-select {
            position: absolute;
            top: 20px;
            right: 20px;
            padding: 10px 15px;
            border: 2px solid #667eea;
            border-radius: 5px;
            font-size: 1em;
            cursor: pointer;
        }
        
        .tabs {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        
        .tab-btn {
            padding: 15px 25px;
            border: 2px solid #667eea;
            background: white;
            color: #667eea;
            font-size: 1em;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: bold;
        }
        
        .tab-btn:hover, .tab-btn.active {
            background: #667eea;
            color: white;
        }
        
        .tab-content {
            display: none;
            animation: fadeIn 0.5s;
        }
        
        .tab-content.active {
            display: block;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        .upload-section {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            margin-bottom: 20px;
        }
        
        input[type="file"] {
            display: none;
        }
        
        .upload-btn {
            background: #667eea;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            margin-top: 10px;
            transition: 0.3s;
        }
        
        .upload-btn:hover {
            background: #764ba2;
        }
        
        .result {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            border-left: 5px solid #667eea;
        }
        
        .sensor-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .sensor-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        
        .sensor-value {
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }
        
        .assistant-input {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        
        .assistant-input input {
            flex: 1;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 1em;
        }
        
        .assistant-input button {
            padding: 12px 30px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
        }
        
        .assistant-response {
            background: #e8eaf6;
            padding: 15px;
            border-radius: 8px;
            color: #333;
            margin-top: 10px;
        }
        
        audio {
            width: 100%;
            margin-top: 15px;
        }
        
        #preview {
            max-width: 100%;
            border-radius: 10px;
            margin-top: 15px;
        }
        
        .loading {
            color: #667eea;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <select class="language-select" id="language">
        <option value="English">English</option>
        <option value="Tamil">Tamil</option>
    </select>
    
    <div class="container">
        <header>
            <h1>🌾 AI Smart Farming Assistant</h1>
            <p class="subtitle">Complete Crop Management System</p>
        </header>
        
        <div class="tabs">
            <button class="tab-btn active" onclick="showTab('disease')">🔍 Disease Detection</button>
            <button class="tab-btn" onclick="showTab('sensors')">📊 Sensors</button>
            <button class="tab-btn" onclick="showTab('assistant')">🤖 Assistant</button>
            <button class="tab-btn" onclick="showTab('irrigation')">💧 Irrigation</button>
        </div>
        
        <!-- Disease Detection -->
        <div id="disease" class="tab-content active">
            <div class="upload-section">
                <h2>📸 Upload Crop Image</h2>
                <input type="file" id="imageInput" accept="image/*">
                <button class="upload-btn" onclick="document.getElementById('imageInput').click()">
                    📤 Choose Image
                </button>
            </div>
            <img id="preview" style="display:none;">
            <button class="upload-btn" onclick="detectDisease()" style="width: 100%; margin-top: 15px;">
                🔎 Detect Disease
            </button>
            <div id="diseaseResult" class="result" style="display:none;"></div>
            <audio id="voiceOutput" controls style="display:none;"></audio>
        </div>
        
        <!-- Sensors -->
        <div id="sensors" class="tab-content">
            <h2>📊 Real-time Sensor Data</h2>
            <div class="sensor-grid">
                <div class="sensor-card">
                    <h3>🌡️ Temperature</h3>
                    <div class="sensor-value" id="temp">28.5°C</div>
                </div>
                <div class="sensor-card">
                    <h3>💧 Humidity</h3>
                    <div class="sensor-value" id="humidity">65%</div>
                </div>
                <div class="sensor-card">
                    <h3>🌱 Soil</h3>
                    <div class="sensor-value" id="soil">45%</div>
                </div>
                <div class="sensor-card">
                    <h3>☀️ Light</h3>
                    <div class="sensor-value" id="light">850 lux</div>
                </div>
            </div>
            <button class="upload-btn" onclick="refreshSensors()" style="width: 100%;">
                🔄 Refresh Data
            </button>
        </div>
        
        <!-- Assistant -->
        <div id="assistant" class="tab-content">
            <h2>🤖 Farming Assistant</h2>
            <div class="assistant-input">
                <input type="text" id="question" placeholder="Ask about farming...">
                <button onclick="askAssistant()">Ask</button>
            </div>
            <div id="assistantResponse" class="assistant-response" style="display:none;"></div>
        </div>
        
        <!-- Irrigation -->
        <div id="irrigation" class="tab-content">
            <h2>💧 Irrigation Control</h2>
            <div style="background: #f8f9fa; padding: 20px; border-radius: 10px;">
                <label style="font-size: 1.1em;">
                    <input type="checkbox" id="autoIrrigation">
                    🔄 Auto Irrigation
                </label>
                <h3 style="margin-top: 20px;">Pump: <span id="pumpStatus" style="color: #667eea;">✅ OFF</span></h3>
            </div>
        </div>
    </div>
    
    <script>
        function showTab(tabName) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }
        
        document.getElementById('imageInput').addEventListener('change', function(e) {
            const reader = new FileReader();
            reader.onload = function(event) {
                document.getElementById('preview').src = event.target.result;
                document.getElementById('preview').style.display = 'block';
            };
            reader.readAsDataURL(e.target.files[0]);
        });
        
        async function detectDisease() {
            const file = document.getElementById('imageInput').files[0];
            if (!file) {
                alert('Please select an image');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', file);
            formData.append('language', document.getElementById('language').value);
            
            try {
                const response = await fetch('/api/detect', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                document.getElementById('diseaseResult').innerHTML = data.result.replace(/\n/g, '<br>');
                document.getElementById('diseaseResult').style.display = 'block';
                
                if (data.audio) {
                    document.getElementById('voiceOutput').src = 'data:audio/mp3;base64,' + data.audio;
                    document.getElementById('voiceOutput').style.display = 'block';
                }
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }
        
        async function refreshSensors() {
            try {
                const response = await fetch('/api/sensors');
                const data = await response.json();
                document.getElementById('temp').textContent = data.temperature + '°C';
                document.getElementById('humidity').textContent = data.humidity + '%';
                document.getElementById('soil').textContent = data.soil_moisture + '%';
                document.getElementById('light').textContent = data.light + ' lux';
            } catch (error) {
                alert('Error refreshing sensors');
            }
        }
        
        async function askAssistant() {
            const q = document.getElementById('question').value;
            if (!q) return;
            
            try {
                const response = await fetch('/api/assistant', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        question: q,
                        language: document.getElementById('language').value
                    })
                });
                const data = await response.json();
                document.getElementById('assistantResponse').innerHTML = data.response.replace(/\n/g, '<br>');
                document.getElementById('assistantResponse').style.display = 'block';
            } catch (error) {
                alert('Error');
            }
        }
        
        refreshSensors();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/detect', methods=['POST'])
def detect():
    file = request.files['file']
    language = request.form.get('language', 'English')
    
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
    lang_code = "ta" if language == "Tamil" else "en"
    remedy = get_remedy(predicted_class, lang_code)
    
    result = f"🌾 Disease: {predicted_class}\n\n📊 Confidence: {confidence:.1f}%\n\n💡 Remedy:\n{remedy}"
    
    audio = None
    try:
        tts = gTTS(text=remedy, lang='ta' if language == "Tamil" else 'en')
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        audio = base64.b64encode(audio_fp.read()).decode()
    except:
        pass
    
    return jsonify({'result': result, 'audio': audio})

@app.route('/api/sensors')
def get_sensors():
    return jsonify(sensor_data)

@app.route('/api/assistant', methods=['POST'])
def assistant():
    data = request.json
    q = data.get('question', '').lower()
    
    advice = {
        "water": "Water your crops in early morning or evening",
        "fertilizer": "Use organic fertilizer every 2 weeks",
        "pest": "Use neem oil spray for pests",
        "disease": "Remove affected leaves immediately",
        "harvest": "Harvest tomatoes when fully ripe"
    }
    
    for key, val in advice.items():
        if key in q:
            return jsonify({'response': f'🤖 {val}'})
    
    return jsonify({'response': '🤖 Ask about: water, fertilizer, pest, disease, harvest'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=True)
