"""
Farmie - AI Crop Care Assistant
A Flask app that combines crop disease detection (TFLite), a bilingual
(English/Tamil) rule-based farming assistant, and a live farm sensor
dashboard (temperature / humidity / soil moisture / irrigation).

Run:
    pip install -r requirements.txt
    python app.py
Then open http://localhost:5000
"""

import base64
import concurrent.futures
import io
import json
import random
import time

import numpy as np
from flask import Flask, jsonify, render_template_string, request
from gtts import gTTS
from PIL import Image

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Model loading (graceful fallback so the whole app still runs for UI/testing
# even if the real .tflite model / labels.json aren't present yet)
# ---------------------------------------------------------------------------
MODEL_READY = False
interpreter = None
input_details = None
output_details = None
labels = []

try:
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path="crop_disease_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    with open("labels.json") as f:
        labels = json.load(f)
    MODEL_READY = True
    print("[Farmie] Real crop disease model loaded.")
except Exception as e:
    print(f"[Farmie] Model not loaded ({e}). Running in DEMO mode with mock predictions.")
    labels = ["Healthy", "Leaf Blight", "Powdery Mildew", "Bacterial Spot", "Leaf Rust"]

# ---------------------------------------------------------------------------
# Remedies (bilingual). Falls back to this if remedies.py isn't provided.
# ---------------------------------------------------------------------------
try:
    from remedies import get_remedy as _external_get_remedy

    def get_remedy(disease, lang):
        return _external_get_remedy(disease, lang)

except ImportError:
    REMEDIES = {
        "Leaf Blight": {
            "en": "Remove and destroy infected leaves. Apply a copper-based fungicide "
                  "every 7-10 days. Avoid overhead watering and improve air circulation "
                  "between plants.",
            "ta": "பாதிக்கப்பட்ட இலைகளை அகற்றி அழிக்கவும். 7-10 நாட்களுக்கு ஒருமுறை "
                  "காப்பர் அடிப்படையிலான பூஞ்சைக் கொல்லி பயன்படுத்தவும். மேலிருந்து நீர் "
                  "பாய்ச்சுவதை தவிர்த்து, செடிகளுக்கு இடையே காற்றோட்டத்தை மேம்படுத்தவும்.",
        },
        "Powdery Mildew": {
            "en": "Spray a mixture of neem oil and water weekly. Ensure good sunlight "
                  "exposure and avoid overcrowding plants.",
            "ta": "வேப்பெண்ணெய் மற்றும் தண்ணீர் கலவையை வாராந்திரம் தெளிக்கவும். "
                  "நல்ல சூரிய ஒளி கிடைப்பதை உறுதி செய்து, செடிகளை நெருக்கமாக "
                  "வளர்ப்பதை தவிர்க்கவும்.",
        },
        "Bacterial Spot": {
            "en": "Use certified disease-free seeds. Apply copper-based bactericides "
                  "and rotate crops each season to reduce bacterial buildup in soil.",
            "ta": "நோய் இல்லாத சான்றளிக்கப்பட்ட விதைகளை பயன்படுத்தவும். காப்பர் "
                  "அடிப்படையிலான பாக்டீரியா நாசினியை தெளிக்கவும், மண்ணில் பாக்டீரியா "
                  "சேராமல் இருக்க ஒவ்வொரு பருவத்திலும் பயிர் சுழற்சி செய்யவும்.",
        },
        "Leaf Rust": {
            "en": "Apply sulfur or triazole-based fungicide at early signs. Remove "
                  "heavily infected leaves and avoid excess nitrogen fertilizer.",
            "ta": "ஆரம்ப அறிகுறிகளில் சல்பர் அல்லது ட்ரையசோல் அடிப்படையிலான "
                  "பூஞ்சைக் கொல்லியை பயன்படுத்தவும். அதிகம் பாதிக்கப்பட்ட இலைகளை "
                  "அகற்றி, அதிகப்படியான நைட்ரஜன் உரத்தை தவிர்க்கவும்.",
        },
    }

    def get_remedy(disease, lang):
        entry = REMEDIES.get(disease)
        if not entry:
            return (
                "Consult your local agricultural extension officer for a precise treatment plan."
                if lang == "en"
                else "துல்லியமான சிகிச்சை திட்டத்திற்கு உங்கள் உள்ளூர் வேளாண் அலுவலரை அணுகவும்."
            )
        return entry.get(lang, entry.get("en", ""))

# ---------------------------------------------------------------------------
# Simulated sensor state (swap this section for real sensor/IoT input later,
# e.g. reading from an MQTT broker, a serial port, or a REST endpoint from
# an ESP32/Arduino device)
# ---------------------------------------------------------------------------
sensor_state = {
    "temperature_c": 28.5,
    "humidity_pct": 62.0,
    "soil_moisture_pct": 41.0,
    "irrigation_on": False,
    "last_updated": time.time(),
}


def refresh_sensors():
    """Simulate a small realistic random walk in sensor readings."""
    sensor_state["temperature_c"] = round(
        min(45, max(10, sensor_state["temperature_c"] + random.uniform(-0.8, 0.8))), 1
    )
    sensor_state["humidity_pct"] = round(
        min(100, max(10, sensor_state["humidity_pct"] + random.uniform(-3, 3))), 1
    )
    # If irrigation is on, soil moisture trends up; otherwise it slowly dries out
    drift = random.uniform(0.5, 2.0) if sensor_state["irrigation_on"] else random.uniform(-1.5, -0.2)
    sensor_state["soil_moisture_pct"] = round(
        min(100, max(0, sensor_state["soil_moisture_pct"] + drift)), 1
    )
    sensor_state["last_updated"] = time.time()
    return sensor_state


# ---------------------------------------------------------------------------
# Bilingual rule-based farming assistant ("Ask Farmie")
# ---------------------------------------------------------------------------
def answer_query(message, lang):
    """A lightweight keyword-matched assistant that gives useful farming
    guidance and can reference live sensor data. No external API needed."""
    text = message.lower()
    s = sensor_state

    def r(en, ta):
        return en if lang == "en" else ta

    # Sensor-aware answers
    if any(w in text for w in ["water", "irrigat", "நீர்", "பாசனம்"]):
        if s["soil_moisture_pct"] < 35:
            return r(
                f"Your soil moisture is currently {s['soil_moisture_pct']}%, which is on the "
                f"drier side. I'd recommend turning on irrigation soon — you can do that from "
                f"the Farm Sensors tab.",
                f"உங்கள் மண் ஈரப்பதம் தற்போது {s['soil_moisture_pct']}% ஆக உள்ளது, இது "
                f"சற்று உலர்ந்த நிலை. Farm Sensors தாவலில் இருந்து பாசனத்தை இயக்குமாறு "
                f"பரிந்துரைக்கிறேன்.",
            )
        return r(
            f"Soil moisture looks healthy right now at {s['soil_moisture_pct']}%. No need to "
            f"irrigate immediately — check again in a few hours.",
            f"தற்போது மண் ஈரப்பதம் {s['soil_moisture_pct']}% ஆக நல்ல நிலையில் உள்ளது. "
            f"உடனடியாக பாசனம் தேவையில்லை — சில மணிநேரங்களில் மீண்டும் சரிபார்க்கவும்.",
        )

    if any(w in text for w in ["temperature", "hot", "cold", "வெப்ப", "வெப்பநிலை"]):
        return r(
            f"Current field temperature is {s['temperature_c']}°C. "
            + ("This is quite warm — make sure crops get enough water and some shade during peak afternoon hours."
               if s["temperature_c"] > 35 else "This is a good range for most crops."),
            f"தற்போதைய வயல் வெப்பநிலை {s['temperature_c']}°C. "
            + ("இது மிகவும் வெப்பமானது — மதிய நேரத்தில் பயிர்களுக்கு போதுமான நீர் மற்றும் நிழல் கிடைக்கிறதா என்பதை உறுதி செய்யவும்."
               if s["temperature_c"] > 35 else "இது பெரும்பாலான பயிர்களுக்கு நல்ல வெப்பநிலை."),
        )

    if any(w in text for w in ["humid", "ஈரப்பதம்"]):
        return r(
            f"Humidity is currently {s['humidity_pct']}%. "
            + ("High humidity can encourage fungal diseases — watch your leaves closely."
               if s["humidity_pct"] > 75 else "This is a normal range."),
            f"தற்போதைய ஈரப்பதம் {s['humidity_pct']}%. "
            + ("அதிக ஈரப்பதம் பூஞ்சை நோய்களை ஊக்குவிக்கலாம் — இலைகளை கவனமாக கண்காணிக்கவும்."
               if s["humidity_pct"] > 75 else "இது இயல்பான அளவு."),
        )

    if any(w in text for w in ["fertiliz", "manure", "உரம்"]):
        return r(
            "For most vegetable crops, a balanced NPK fertilizer (like 19-19-19) works well "
            "during vegetative growth, switching to higher phosphorus/potassium during "
            "flowering and fruiting. Composted organic manure improves long-term soil health.",
            "பெரும்பாலான காய்கறி பயிர்களுக்கு, வளர்ச்சி கட்டத்தில் சமச்சீர் NPK உரம் "
            "(19-19-19 போன்றவை) நன்றாக வேலை செய்யும். பூக்கும்/காய்க்கும் நேரத்தில் "
            "அதிக பாஸ்பரஸ்/பொட்டாசியம் கொண்ட உரத்திற்கு மாறவும். மக்கிய இயற்கை உரம் "
            "நீண்ட கால மண் ஆரோக்கியத்தை மேம்படுத்தும்.",
        )

    if any(w in text for w in ["pest", "insect", "bug", "பூச்சி"]):
        return r(
            "For general pest control, try neem oil spray as a first, low-toxicity option. "
            "For severe infestations, identify the specific pest before choosing a targeted "
            "pesticide — overuse of broad pesticides harms beneficial insects too.",
            "பொதுவான பூச்சி கட்டுப்பாட்டிற்கு, குறைந்த நச்சுத்தன்மை கொண்ட வேப்பெண்ணெய் "
            "தெளிப்பை முதலில் முயற்சிக்கவும். கடுமையான பூச்சி தாக்குதலுக்கு, குறிப்பிட்ட "
            "பூச்சியை அடையாளம் கண்ட பின் மருந்தை தேர்வு செய்யவும் — பரவலான பூச்சிக்கொல்லிகள் "
            "நன்மை பயக்கும் பூச்சிகளையும் பாதிக்கும்.",
        )

    if any(w in text for w in ["disease", "leaf", "spot", "blight", "rust", "mildew", "நோய்", "இலை"]):
        return r(
            "For disease diagnosis, upload a clear photo of the affected leaf in the "
            "'Detect Disease' tab and I'll analyze it and suggest a specific remedy.",
            "நோய் கண்டறிதலுக்கு, 'Detect Disease' தாவலில் பாதிக்கப்பட்ட இலையின் தெளிவான "
            "புகைப்படத்தை பதிவேற்றவும், நான் அதை பகுப்பாய்வு செய்து குறிப்பிட்ட தீர்வை "
            "பரிந்துரைக்கிறேன்.",
        )

    if any(w in text for w in ["hello", "hi", "வணக்கம்"]):
        return r(
            "Hello! I'm Farmie 🌱 Ask me about watering, fertilizer, pests, or upload a "
            "photo to check for crop disease.",
            "வணக்கம்! நான் ஃபார்மி 🌱 நீர்ப்பாசனம், உரம், பூச்சிகள் பற்றி கேளுங்கள், "
            "அல்லது பயிர் நோயை சரிபார்க்க ஒரு புகைப்படத்தை பதிவேற்றவும்.",
        )

    # Fallback
    return r(
        "I can help with watering schedules, fertilizer advice, pest control, and crop "
        "disease detection. Could you tell me a bit more about what you'd like help with?",
        "நீர்ப்பாசன அட்டவணை, உர ஆலோசனை, பூச்சி கட்டுப்பாடு மற்றும் பயிர் நோய் "
        "கண்டறிதலுக்கு நான் உதவ முடியும். நீங்கள் எதற்கு உதவி வேண்டும் என்பதை "
        "இன்னும் கொஞ்சம் சொல்ல முடியுமா?",
    )


# ---------------------------------------------------------------------------
# HTML / CSS / JS
# ---------------------------------------------------------------------------
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Farmie - AI Crop Care</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Noto+Sans+Tamil:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --primary: #2f9e5b;
    --primary-dark: #1f7a44;
    --primary-light: #eaf7ef;
    --accent: #ffb020;
    --bg: #f4f7f5;
    --card: #ffffff;
    --text: #1e2b23;
    --muted: #6b7d72;
    --danger: #e2574c;
    --radius: 18px;
    --shadow: 0 10px 30px rgba(31, 122, 68, 0.10);
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Poppins', 'Noto Sans Tamil', sans-serif;
    background: linear-gradient(160deg, #eafaf0 0%, #f4f7f5 40%, #eef6fb 100%);
    color: var(--text);
    min-height: 100vh;
    padding: 24px 16px 60px;
  }
  .shell { max-width: 980px; margin: 0 auto; }

  header.top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 24px;
    flex-wrap: wrap;
    gap: 12px;
  }
  .brand { display: flex; align-items: center; gap: 14px; }
  .brand .avatar {
    font-size: 42px;
    background: var(--primary-light);
    border-radius: 16px;
    width: 64px; height: 64px;
    display: flex; align-items: center; justify-content: center;
    box-shadow: var(--shadow);
  }
  .brand h1 { font-size: 1.7em; font-weight: 800; color: var(--primary-dark); }
  .brand p { color: var(--muted); font-size: 0.95em; }

  .lang-switch {
    display: flex;
    background: var(--card);
    border-radius: 30px;
    padding: 4px;
    box-shadow: var(--shadow);
  }
  .lang-switch button {
    border: none;
    background: transparent;
    padding: 9px 18px;
    border-radius: 30px;
    font-weight: 600;
    font-size: 0.9em;
    cursor: pointer;
    color: var(--muted);
    transition: all .25s;
  }
  .lang-switch button.active {
    background: var(--primary);
    color: white;
  }

  .tabs {
    display: flex;
    gap: 8px;
    margin-bottom: 22px;
    background: var(--card);
    padding: 6px;
    border-radius: 16px;
    box-shadow: var(--shadow);
    overflow-x: auto;
  }
  .tab-btn {
    flex: 1;
    min-width: 140px;
    padding: 13px 10px;
    border: none;
    background: transparent;
    border-radius: 12px;
    font-weight: 600;
    font-size: 0.95em;
    color: var(--muted);
    cursor: pointer;
    transition: all .2s;
    white-space: nowrap;
  }
  .tab-btn.active { background: var(--primary); color: white; }

  .panel { display: none; }
  .panel.active { display: block; animation: fadeIn .35s ease; }
  @keyframes fadeIn { from{opacity:0; transform:translateY(8px);} to{opacity:1; transform:translateY(0);} }

  .card {
    background: var(--card);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    padding: 28px;
    margin-bottom: 20px;
  }

  .upload-box {
    border: 2.5px dashed #b7e0c6;
    border-radius: var(--radius);
    padding: 36px 24px;
    text-align: center;
    background: var(--primary-light);
  }
  .upload-box h2 { font-size: 1.2em; margin-bottom: 8px; color: var(--primary-dark); }
  .upload-box p { color: var(--muted); margin-bottom: 18px; }
  #imageInput { display: none; }

  .btn {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
    color: white;
    padding: 14px 32px;
    border: none;
    border-radius: 14px;
    font-size: 1em;
    font-weight: 700;
    cursor: pointer;
    transition: all .2s;
    box-shadow: 0 8px 20px rgba(47, 158, 91, 0.28);
  }
  .btn:hover { transform: translateY(-2px); box-shadow: 0 12px 26px rgba(47, 158, 91, 0.4); }
  .btn:disabled { opacity: 0.55; cursor: not-allowed; transform: none; }
  .btn.secondary {
    background: white;
    color: var(--primary-dark);
    border: 2px solid var(--primary);
    box-shadow: none;
  }
  .btn.block { width: 100%; margin-top: 18px; padding: 16px; font-size: 1.05em; }

  #imagePreview { display:none; max-width: 320px; width: 100%; border-radius: 16px; margin: 20px auto 0; display:block; }

  .loading { display: none; text-align: center; margin: 22px 0; }
  .spinner {
    border: 4px solid #e4f4ea;
    border-top: 4px solid var(--primary);
    border-radius: 50%;
    width: 36px; height: 36px;
    animation: spin 0.9s linear infinite;
    margin: 0 auto 10px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .result { display: none; margin-top: 22px; }
  .result.show { display: block; animation: fadeIn .4s ease; }
  .result-card {
    border-radius: 16px;
    padding: 22px;
    border-left: 5px solid var(--primary);
    background: var(--primary-light);
  }
  .result-card.warning { border-left-color: var(--danger); background: #fdecea; }
  .result-title { font-size: 1.15em; font-weight: 700; margin-bottom: 10px; }
  .result-text { white-space: pre-wrap; line-height: 1.75; color: var(--text); }
  .farmie-msg {
    background: white; border-radius: 14px; padding: 16px 18px;
    margin-top: 16px; font-size: 0.98em; border: 1.5px solid #dcefe2;
  }
  audio { width: 100%; margin-top: 16px; border-radius: 10px; }

  /* Chat */
  .chat-window {
    height: 420px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 12px;
    padding: 6px 4px 12px;
  }
  .msg { max-width: 78%; padding: 12px 16px; border-radius: 16px; line-height: 1.55; font-size: 0.97em; }
  .msg.bot { align-self: flex-start; background: var(--primary-light); border-bottom-left-radius: 4px; }
  .msg.user { align-self: flex-end; background: var(--primary); color: white; border-bottom-right-radius: 4px; }
  .msg .speak-btn {
    display: inline-block; margin-top: 8px; font-size: 0.82em; cursor: pointer;
    color: var(--primary-dark); font-weight: 600; background:none; border:none; padding:0;
  }
  .msg.user .speak-btn { color: #eafff0; }

  .chat-input-row { display: flex; gap: 10px; margin-top: 16px; }
  .chat-input-row input[type=text] {
    flex: 1;
    padding: 14px 16px;
    border-radius: 14px;
    border: 2px solid #e1ede4;
    font-size: 1em;
    font-family: inherit;
    outline: none;
  }
  .chat-input-row input[type=text]:focus { border-color: var(--primary); }
  .icon-btn {
    width: 50px; height: 50px; border-radius: 14px; border: none;
    background: var(--primary-light); color: var(--primary-dark);
    font-size: 1.2em; cursor: pointer; flex-shrink: 0;
    display:flex; align-items:center; justify-content:center;
  }
  .icon-btn.send { background: var(--primary); color: white; }
  .icon-btn:disabled { opacity: 0.5; cursor: not-allowed; }

  /* Sensors */
  .sensor-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 16px;
    margin-bottom: 18px;
  }
  .sensor-card {
    background: var(--primary-light);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
  }
  .sensor-card .icon { font-size: 1.8em; margin-bottom: 6px; }
  .sensor-card .value { font-size: 1.7em; font-weight: 800; color: var(--primary-dark); }
  .sensor-card .label { color: var(--muted); font-size: 0.9em; margin-top: 4px; }
  .sensor-card.on { background: #eafaf0; box-shadow: inset 0 0 0 2px var(--primary); }
  .sensor-actions { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }
  .updated-note { color: var(--muted); font-size: 0.85em; margin-top: 12px; }
  .badge {
    display: inline-block; padding: 4px 12px; border-radius: 20px;
    font-size: 0.8em; font-weight: 700;
  }
  .badge.demo { background: #fff3d9; color: #92650a; }
  .badge.live { background: #dbf5e4; color: var(--primary-dark); }
</style>
</head>
<body>
<div class="shell">

  <header class="top">
    <div class="brand">
      <div class="avatar">🌾</div>
      <div>
        <h1 id="brandTitle">Farmie</h1>
        <p id="brandSubtitle">Your AI Crop Care Assistant</p>
      </div>
    </div>
    <div class="lang-switch">
      <button id="langEnBtn" class="active" onclick="setLang('english')">English</button>
      <button id="langTaBtn" onclick="setLang('tamil')">தமிழ்</button>
    </div>
  </header>

  <div class="tabs">
    <button class="tab-btn active" data-tab="detect" id="tabDetectBtn" onclick="showTab('detect')">📷 <span id="tabDetectLabel">Detect Disease</span></button>
    <button class="tab-btn" data-tab="chat" id="tabChatBtn" onclick="showTab('chat')">💬 <span id="tabChatLabel">Ask Farmie</span></button>
    <button class="tab-btn" data-tab="sensors" id="tabSensorsBtn" onclick="showTab('sensors')">🌡️ <span id="tabSensorsLabel">Farm Sensors</span></button>
  </div>

  <!-- DETECT TAB -->
  <div class="panel active" id="panel-detect">
    <div class="card">
      <div class="upload-box">
        <h2 id="uploadTitle">📸 Upload Your Crop Image</h2>
        <p id="uploadText">Choose a clear photo of the affected leaf</p>
        <input type="file" id="imageInput" accept="image/*">
        <button class="btn" onclick="document.getElementById('imageInput').click()" id="uploadBtn">📤 Choose Image</button>
      </div>

      <img id="imagePreview" alt="Preview" style="display:none;">
      <button class="btn block" style="display:none;" onclick="detectDisease()" id="detectBtn">🔎 Ask Farmie to Analyze</button>

      <div class="loading" id="loading">
        <div class="spinner"></div>
        <p id="loadingText" style="color: var(--primary-dark); font-weight:600;">Analyzing...</p>
      </div>

      <div class="result" id="result">
        <div class="result-card" id="resultCard">
          <div class="result-title" id="resultTitle"></div>
          <div class="result-text" id="resultText"></div>
        </div>
        <div class="farmie-msg" id="farmieMsg"></div>
        <audio id="audio" controls></audio>
      </div>
    </div>
  </div>

  <!-- CHAT TAB -->
  <div class="panel" id="panel-chat">
    <div class="card">
      <div class="chat-window" id="chatWindow"></div>
      <div class="chat-input-row">
        <button class="icon-btn" id="micBtn" onclick="toggleMic()" title="Speak your question">🎤</button>
        <input type="text" id="chatInput" placeholder="Ask about watering, pests, fertilizer..." onkeydown="if(event.key==='Enter') sendChat();">
        <button class="icon-btn send" onclick="sendChat()" title="Send">➤</button>
      </div>
    </div>
  </div>

  <!-- SENSORS TAB -->
  <div class="panel" id="panel-sensors">
    <div class="card">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px; flex-wrap:wrap; gap:10px;">
        <h2 id="sensorHeading" style="color:var(--primary-dark);">🌡️ Live Farm Conditions</h2>
        <span class="badge demo" id="sensorBadge">DEMO DATA</span>
      </div>
      <div class="sensor-grid" id="sensorGrid"></div>
      <div class="sensor-actions">
        <button class="btn secondary" onclick="refreshSensors()" id="refreshBtn">🔄 <span id="refreshLabel">Refresh</span></button>
        <button class="btn" onclick="toggleIrrigation()" id="irrigationBtn">💧 <span id="irrigationLabel">Turn Irrigation On</span></button>
      </div>
      <p class="updated-note" id="updatedNote"></p>
    </div>
  </div>

</div>

<script>
let lang = 'english';
let mediaRecognition = null;
let micActive = false;

const T = {
  english: {
    brandTitle: "Farmie", brandSubtitle: "Your AI Crop Care Assistant",
    tabDetect: "Detect Disease", tabChat: "Ask Farmie", tabSensors: "Farm Sensors",
    uploadTitle: "📸 Upload Your Crop Image", uploadText: "Choose a clear photo of the affected leaf",
    uploadBtn: "📤 Choose Image", detectBtn: "🔎 Ask Farmie to Analyze", loadingText: "Analyzing...",
    chatPlaceholder: "Ask about watering, pests, fertilizer...",
    sensorHeading: "🌡️ Live Farm Conditions", refreshLabel: "Refresh",
    irrigationOn: "Turn Irrigation Off", irrigationOff: "Turn Irrigation On",
    temp: "Temperature", humidity: "Humidity", soil: "Soil Moisture", irrigation: "Irrigation",
    updated: "Last updated: ", chatWelcome: "Hello! I'm Farmie 🌱 Ask me about watering, fertilizer, pests, or upload a photo to check for crop disease.",
    micNotSupported: "Voice input isn't supported in this browser.", listening: "Listening...",
    selectImage: "Please select an image first.",
  },
  tamil: {
    brandTitle: "ஃபார்மி", brandSubtitle: "உங்கள் AI பயிர் பராமரிப்பு உதவியாளர்",
    tabDetect: "நோய் கண்டறிதல்", tabChat: "ஃபார்மியிடம் கேளுங்கள்", tabSensors: "பண்ணை சென்சார்கள்",
    uploadTitle: "📸 உங்கள் பயிரின் படத்தை பதிவேற்றவும்", uploadText: "பாதிக்கப்பட்ட இலையின் தெளிவான படத்தை தேர்ந்தெடுக்கவும்",
    uploadBtn: "📤 படத்தைத் தேர்ந்தெடுக்கவும்", detectBtn: "🔎 ஃபார்மியிடம் பகுப்பாய்வு கேளுங்கள்", loadingText: "பகுப்பாய்வு செய்கிறது...",
    chatPlaceholder: "நீர்ப்பாசனம், பூச்சி, உரம் பற்றி கேளுங்கள்...",
    sensorHeading: "🌡️ நேரடி பண்ணை நிலைமைகள்", refreshLabel: "புதுப்பிக்க",
    irrigationOn: "பாசனத்தை நிறுத்து", irrigationOff: "பாசனத்தை இயக்கு",
    temp: "வெப்பநிலை", humidity: "ஈரப்பதம்", soil: "மண் ஈரப்பதம்", irrigation: "பாசனம்",
    updated: "கடைசியாக புதுப்பிக்கப்பட்டது: ", chatWelcome: "வணக்கம்! நான் ஃபார்மி 🌱 நீர்ப்பாசனம், உரம், பூச்சிகள் பற்றி கேளுங்கள், அல்லது பயிர் நோயை சரிபார்க்க ஒரு புகைப்படத்தை பதிவேற்றவும்.",
    micNotSupported: "இந்த உலாவியில் குரல் உள்ளீடு ஆதரிக்கப்படவில்லை.", listening: "கேட்கிறது...",
    selectImage: "முதலில் ஒரு படத்தை தேர்ந்தெடுக்கவும்.",
  }
};

function setLang(l) {
  lang = l;
  document.getElementById('langEnBtn').classList.toggle('active', l === 'english');
  document.getElementById('langTaBtn').classList.toggle('active', l === 'tamil');
  const t = T[l];
  document.getElementById('brandTitle').textContent = t.brandTitle;
  document.getElementById('brandSubtitle').textContent = t.brandSubtitle;
  document.getElementById('tabDetectLabel').textContent = t.tabDetect;
  document.getElementById('tabChatLabel').textContent = t.tabChat;
  document.getElementById('tabSensorsLabel').textContent = t.tabSensors;
  document.getElementById('uploadTitle').textContent = t.uploadTitle;
  document.getElementById('uploadText').textContent = t.uploadText;
  document.getElementById('uploadBtn').textContent = t.uploadBtn;
  document.getElementById('detectBtn').textContent = t.detectBtn;
  document.getElementById('loadingText').textContent = t.loadingText;
  document.getElementById('chatInput').placeholder = t.chatPlaceholder;
  document.getElementById('sensorHeading').textContent = t.sensorHeading;
  document.getElementById('refreshLabel').textContent = t.refreshLabel;
  document.getElementById('irrigationLabel').textContent = sensorData && sensorData.irrigation_on ? t.irrigationOn : t.irrigationOff;
  renderSensors();
}

function showTab(name) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('panel-' + name).classList.add('active');
  document.getElementById('tab' + name.charAt(0).toUpperCase() + name.slice(1) + 'Btn').classList.add('active');
  if (name === 'sensors') refreshSensors();
  if (name === 'chat' && !chatStarted) startChat();
}

// ---------- DETECT ----------
document.getElementById('imageInput').addEventListener('change', function (e) {
  const file = e.target.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = function (event) {
      document.getElementById('imagePreview').src = event.target.result;
      document.getElementById('imagePreview').style.display = 'block';
      document.getElementById('detectBtn').style.display = 'block';
    };
    reader.readAsDataURL(file);
  }
});

async function detectDisease() {
  const file = document.getElementById('imageInput').files[0];
  if (!file) { alert(T[lang].selectImage); return; }

  document.getElementById('loading').style.display = 'block';
  document.getElementById('result').classList.remove('show');
  document.getElementById('detectBtn').disabled = true;

  const formData = new FormData();
  formData.append('file', file);
  formData.append('language', lang);

  try {
    const response = await fetch('/api/detect', { method: 'POST', body: formData });
    let data;
    try {
      data = await response.json();
    } catch (parseErr) {
      throw new Error('Server did not return a valid response (HTTP ' + response.status + '). Please try again.');
    }
    if (!response.ok) {
      throw new Error(data.error || ('Server error (HTTP ' + response.status + ')'));
    }

    document.getElementById('resultTitle').textContent = data.title;
    document.getElementById('resultText').textContent = data.result;
    document.getElementById('farmieMsg').textContent = data.farmie_msg;
    document.getElementById('resultCard').classList.toggle('warning', !data.healthy);

    if (data.audio) {
      document.getElementById('audio').src = 'data:audio/mp3;base64,' + data.audio;
    }
    document.getElementById('result').classList.add('show');
  } catch (error) {
    alert('Error: ' + error);
  } finally {
    document.getElementById('loading').style.display = 'none';
    document.getElementById('detectBtn').disabled = false;
  }
}

// ---------- CHAT ----------
let chatStarted = false;
function startChat() {
  chatStarted = true;
  addMessage(T[lang].chatWelcome, 'bot', false);
}

function addMessage(text, who, withSpeak = true) {
  const win = document.getElementById('chatWindow');
  const div = document.createElement('div');
  div.className = 'msg ' + who;
  div.textContent = text;
  if (withSpeak) {
    const speakBtn = document.createElement('button');
    speakBtn.className = 'speak-btn';
    speakBtn.textContent = '🔊';
    speakBtn.onclick = () => speakText(text);
    div.appendChild(document.createElement('br'));
    div.appendChild(speakBtn);
  }
  win.appendChild(div);
  win.scrollTop = win.scrollHeight;
}

async function sendChat() {
  const input = document.getElementById('chatInput');
  const message = input.value.trim();
  if (!message) return;
  addMessage(message, 'user', false);
  input.value = '';

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, language: lang })
    });
    const data = await response.json();
    addMessage(data.reply, 'bot');
  } catch (error) {
    addMessage('Error: ' + error, 'bot', false);
  }
}

async function speakText(text) {
  try {
    const response = await fetch('/api/tts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, language: lang })
    });
    const data = await response.json();
    if (data.audio) {
      const audio = new Audio('data:audio/mp3;base64,' + data.audio);
      audio.play();
    }
  } catch (e) { console.error(e); }
}

// Voice input (Web Speech API) — graceful fallback if unsupported
function toggleMic() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) { alert(T[lang].micNotSupported); return; }

  if (micActive && mediaRecognition) { mediaRecognition.stop(); return; }

  mediaRecognition = new SpeechRecognition();
  mediaRecognition.lang = lang === 'tamil' ? 'ta-IN' : 'en-US';
  mediaRecognition.interimResults = false;
  mediaRecognition.maxAlternatives = 1;

  mediaRecognition.onstart = () => {
    micActive = true;
    document.getElementById('micBtn').style.background = '#ffdada';
    document.getElementById('chatInput').placeholder = T[lang].listening;
  };
  mediaRecognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    document.getElementById('chatInput').value = transcript;
    sendChat();
  };
  mediaRecognition.onerror = () => { micActive = false; document.getElementById('micBtn').style.background = ''; };
  mediaRecognition.onend = () => {
    micActive = false;
    document.getElementById('micBtn').style.background = '';
    document.getElementById('chatInput').placeholder = T[lang].chatPlaceholder;
  };
  mediaRecognition.start();
}

// ---------- SENSORS ----------
let sensorData = null;

function renderSensors() {
  if (!sensorData) return;
  const t = T[lang];
  const grid = document.getElementById('sensorGrid');
  grid.innerHTML = `
    <div class="sensor-card">
      <div class="icon">🌡️</div>
      <div class="value">${sensorData.temperature_c}°C</div>
      <div class="label">${t.temp}</div>
    </div>
    <div class="sensor-card">
      <div class="icon">💦</div>
      <div class="value">${sensorData.humidity_pct}%</div>
      <div class="label">${t.humidity}</div>
    </div>
    <div class="sensor-card">
      <div class="icon">🌱</div>
      <div class="value">${sensorData.soil_moisture_pct}%</div>
      <div class="label">${t.soil}</div>
    </div>
    <div class="sensor-card ${sensorData.irrigation_on ? 'on' : ''}">
      <div class="icon">💧</div>
      <div class="value">${sensorData.irrigation_on ? 'ON' : 'OFF'}</div>
      <div class="label">${t.irrigation}</div>
    </div>
  `;
  document.getElementById('irrigationLabel').textContent = sensorData.irrigation_on ? t.irrigationOn : t.irrigationOff;
  const d = new Date(sensorData.last_updated * 1000);
  document.getElementById('updatedNote').textContent = t.updated + d.toLocaleTimeString();
}

async function refreshSensors() {
  const btn = document.getElementById('refreshBtn');
  btn.disabled = true;
  try {
    const response = await fetch('/api/sensors');
    sensorData = await response.json();
    renderSensors();
  } catch (e) { console.error(e); }
  finally { btn.disabled = false; }
}

async function toggleIrrigation() {
  const btn = document.getElementById('irrigationBtn');
  btn.disabled = true;
  try {
    const response = await fetch('/api/sensors', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'toggle_irrigation' })
    });
    sensorData = await response.json();
    renderSensors();
  } catch (e) { console.error(e); }
  finally { btn.disabled = false; }
}

// init
refreshSensors();
setInterval(() => { if (document.getElementById('panel-sensors').classList.contains('active')) refreshSensors(); }, 15000);
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/detect", methods=["POST"])
def detect():
    try:
        return _run_detect()
    except Exception as e:
        # Guarantee a JSON response even on an unexpected failure, so the
        # front-end never tries to parse an HTML error page as JSON.
        print("Detect Error:", e)
        return jsonify({"error": f"Detection failed: {e}"}), 500


def _run_detect():
    file = request.files["file"]
    language = request.form.get("language", "english")
    lang_code = "ta" if language == "tamil" else "en"

    image = Image.open(file).convert("RGB").resize((224, 224))

    if MODEL_READY:
        input_data = np.array(image, dtype=np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])
        predicted_class = labels[int(np.argmax(output[0]))]
        confidence = float(np.max(output[0])) * 100
    else:
        # DEMO mode: no real model file present, return a plausible mock result
        predicted_class = random.choice(labels)
        confidence = round(random.uniform(78, 97), 1)

    healthy = "healthy" in predicted_class.lower()

    if healthy:
        if language == "tamil":
            title = "✅ வாழ்த்துக்கள்! ஆரோக்கியமாக உள்ளது!"
            result = (
                f"🌿 நிலை: ஆரோக்கியமான பயிர்\n"
                f"📊 நம்பிக்கை: {confidence:.1f}%\n\n"
                f"உங்கள் பயிர் ஆரோக்கியமாக உள்ளது."
            )
            farmie_msg = "🌱 ஃபார்மி: உங்கள் பயிர் ஆரோக்கியமாக இருக்கிறது! தொடர்ந்து நல்ல பராமரிப்பை செய்யுங்கள்."
        else:
            title = "✅ Great! Your Crop is Healthy!"
            result = f"🌿 Status: HEALTHY\n📊 Confidence: {confidence:.1f}%\n\nYour crop appears to be healthy."
            farmie_msg = "🌱 Farmie: Your crop looks healthy! Keep monitoring it regularly."
    else:
        remedy = get_remedy(predicted_class, lang_code)
        if language == "tamil":
            title = f"⚠️ பயிர் நோய் கண்டறியப்பட்டது: {predicted_class}"
            result = (
                f"🌿 நோய்: {predicted_class}\n"
                f"📊 நம்பிக்கை: {confidence:.1f}%\n\n"
                f"💊 பரிந்துரைக்கப்பட்ட தீர்வு:\n{remedy}"
            )
            farmie_msg = f"🌱 ஃபார்மி: உங்கள் பயிரில் {predicted_class} கண்டறியப்பட்டுள்ளது."
        else:
            title = f"⚠️ Crop Disease Detected: {predicted_class}"
            result = f"🌿 Disease: {predicted_class}\n📊 Confidence: {confidence:.1f}%\n\n💊 Recommended Remedy:\n{remedy}"
            farmie_msg = f"🌱 Farmie: {predicted_class} has been detected in your crop."

    audio_base64 = _tts_base64(farmie_msg, lang_code)

    return jsonify(
        {
            "title": title,
            "result": result,
            "farmie_msg": farmie_msg,
            "audio": audio_base64,
            "healthy": healthy,
            "demo_mode": not MODEL_READY,
        }
    )


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    message = data.get("message", "")
    language = data.get("language", "english")
    reply = answer_query(message, "en" if language == "english" else "ta")
    return jsonify({"reply": reply})


@app.route("/api/tts", methods=["POST"])
def tts():
    data = request.get_json(force=True)
    text = data.get("text", "")
    language = data.get("language", "english")
    lang_code = "ta" if language == "tamil" else "en"
    audio_base64 = _tts_base64(text, lang_code)
    return jsonify({"audio": audio_base64})


@app.route("/api/sensors", methods=["GET", "POST"])
def sensors():
    if request.method == "POST":
        data = request.get_json(force=True)
        if data.get("action") == "toggle_irrigation":
            sensor_state["irrigation_on"] = not sensor_state["irrigation_on"]
    refresh_sensors()
    return jsonify(sensor_state)


def _tts_base64(text, lang_code, timeout=8):
    if not text:
        return None

    def _generate():
        tts_obj = gTTS(text=text, lang=lang_code)
        buf = io.BytesIO()
        tts_obj.write_to_fp(buf)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    # gTTS calls out to Google Translate over the network. On some hosts
    # (e.g. Render) that call can hang or be blocked, which was stalling
    # the whole /api/detect request until the server timed out and
    # returned an HTML error page instead of JSON. Running it with a hard
    # timeout means a slow/blocked TTS call just gets skipped (no audio)
    # instead of breaking the whole response.
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_generate)
            return future.result(timeout=timeout)
    except Exception as e:
        print("TTS Error:", e)
        return None


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
