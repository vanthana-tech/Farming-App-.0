"""
Farmie - AI Crop Care Assistant
Features:
- AI crop disease detection using TFLite
- English / Tamil support
- Ask Farmie farming assistant
- Live temperature / humidity / soil / irrigation status
- ESP32 → Render sensor updates
- Website → Render → ESP32 irrigation ON/OFF command
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

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    print("[Farmie] HEIC/HEIF support enabled.")
except ImportError:
    print("[Farmie] pillow-heif not installed.")

app = Flask(__name__)

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

    with open("labels.json", "r", encoding="utf-8") as f:
        labels_raw = json.load(f)

    if isinstance(labels_raw, dict):
        labels = [labels_raw[k] for k in sorted(labels_raw, key=lambda x: int(x))]
    else:
        labels = labels_raw

    MODEL_READY = True
    print("[Farmie] Real crop disease model loaded.")
except Exception as e:
    print(f"[Farmie] Model not loaded: {e}")
    print("[Farmie] Running in DEMO mode.")
    labels = [
        "Healthy",
        "Leaf Blight",
        "Powdery Mildew",
        "Bacterial Spot",
        "Leaf Rust"
    ]

try:
    from remedies import get_remedy as _external_get_remedy

    def get_remedy(disease, lang):
        return _external_get_remedy(disease, lang)

except ImportError:
    REMEDIES = {
        "Leaf Blight": {
            "en": "Remove and destroy infected leaves. Avoid overhead watering and improve air circulation.",
            "ta": "பாதிக்கப்பட்ட இலைகளை அகற்றி அழிக்கவும். மேலிருந்து நீர் பாய்ச்சுவதை தவிர்த்து, காற்றோட்டத்தை மேம்படுத்தவும்."
        },
        "Powdery Mildew": {
            "en": "Use neem oil spray as a general treatment. Ensure good sunlight and avoid overcrowding.",
            "ta": "வேப்பெண்ணெய் தெளிப்பை பயன்படுத்தவும். நல்ல சூரிய ஒளி கிடைப்பதை உறுதி செய்யவும்."
        },
        "Bacterial Spot": {
            "en": "Use disease-free seeds and maintain good field hygiene. Consult an agricultural expert for severe infection.",
            "ta": "நோய் இல்லாத விதைகளை பயன்படுத்தவும். கடுமையான பாதிப்பிற்கு வேளாண் நிபுணரை அணுகவும்."
        },
        "Leaf Rust": {
            "en": "Remove heavily infected leaves and improve air circulation.",
            "ta": "அதிகம் பாதிக்கப்பட்ட இலைகளை அகற்றி காற்றோட்டத்தை மேம்படுத்தவும்."
        }
    }

    def get_remedy(disease, lang):
        entry = REMEDIES.get(disease)
        if not entry:
            return "Consult your local agricultural extension officer." if lang == "en" else "உங்கள் உள்ளூர் வேளாண் அலுவலரை அணுகவும்."
        return entry.get(lang, entry.get("en", ""))

sensor_state = {
    "temperature_c": None,
    "humidity_pct": None,
    "soil_status": "UNKNOWN",
    "irrigation_on": False,
    "last_updated": 0
}

irrigation_command = False
command_updated = 0

def answer_query(message, lang):
    text = message.lower()
    s = sensor_state

    def r(en, ta):
        return en if lang == "en" else ta

    if any(w in text for w in ["water", "irrigat", "நீர்", "பாசனம்"]):
        if s["soil_status"] == "DRY":
            return r(
                "The soil is currently DRY. I recommend irrigation. You can turn the pump ON from the Farm Sensors tab.",
                "மண் தற்போது உலர்ந்த நிலையில் உள்ளது. பாசனம் செய்ய பரிந்துரைக்கிறேன். Farm Sensors பகுதியில் இருந்து பம்பை ON செய்யலாம்."
            )
        return r(
            "The soil is currently WET. Irrigation may not be necessary right now.",
            "மண் தற்போது ஈரமாக உள்ளது. இப்போது பாசனம் தேவையில்லை."
        )

    if any(w in text for w in ["temperature", "hot", "cold", "வெப்ப", "வெப்பநிலை"]):
        temp = s["temperature_c"]
        if temp is None:
            return r(
                "Temperature data is not available yet.",
                "வெப்பநிலை தகவல் இன்னும் கிடைக்கவில்லை."
            )
        return r(
            f"Current temperature is {temp}°C.",
            f"தற்போதைய வெப்பநிலை {temp}°C."
        )

    if any(w in text for w in ["humid", "ஈரப்பதம்"]):
        hum = s["humidity_pct"]
        if hum is None:
            return r(
                "Humidity data is not available yet.",
                "ஈரப்பதம் தகவல் இன்னும் கிடைக்கவில்லை."
            )
        return r(
            f"Current humidity is {hum}%.",
            f"தற்போதைய ஈரப்பதம் {hum}%."
        )

    if any(w in text for w in ["fertiliz", "manure", "உரம்"]):
        return r(
            "Use a balanced fertilizer according to your crop's growth stage. Organic manure can also improve soil health.",
            "பயிரின் வளர்ச்சி நிலைக்கு ஏற்ப சமச்சீர் உரத்தை பயன்படுத்தவும். இயற்கை உரம் மண் ஆரோக்கியத்தை மேம்படுத்தும்."
        )

    if any(w in text for w in ["pest", "insect", "bug", "பூச்சி"]):
        return r(
            "For general pest control, neem oil can be considered. For severe infestation, identify the pest before using pesticides.",
            "பொதுவான பூச்சி கட்டுப்பாட்டிற்கு வேப்பெண்ணெய் பயன்படுத்தலாம். கடுமையான தாக்குதலில் பூச்சியை அடையாளம் கண்ட பிறகு மருந்தை பயன்படுத்தவும்."
        )

    if any(w in text for w in ["disease", "leaf", "spot", "blight", "rust", "mildew", "நோய்", "இலை"]):
        return r(
            "Upload a clear crop leaf photo in the Detect Disease tab.",
            "Detect Disease பகுதியில் பயிரின் இலையின் தெளிவான புகைப்படத்தை பதிவேற்றவும்."
        )

    if any(w in text for w in ["hello", "hi", "வணக்கம்"]):
        return r(
            "Hello! I'm Farmie 🌱 Ask me about watering, fertilizer, pests or crop disease.",
            "வணக்கம்! நான் Farmie 🌱 நீர்ப்பாசனம், உரம், பூச்சி அல்லது பயிர் நோய் பற்றி கேளுங்கள்."
        )

    return r(
        "I can help with watering, fertilizer, pests and crop disease detection.",
        "நீர்ப்பாசனம், உரம், பூச்சி மற்றும் பயிர் நோய் கண்டறிதலில் நான் உதவ முடியும்."
    )

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
:root{--primary:#2f9e5b;--primary-dark:#1f7a44;--primary-light:#eaf7ef;--accent:#ffb020;--bg:#f4f7f5;--card:#ffffff;--text:#1e2b23;--muted:#6b7d72;--danger:#e2574c;--radius:18px;--shadow:0 10px 30px rgba(31,122,68,.10)}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Poppins','Noto Sans Tamil',sans-serif;background:linear-gradient(160deg,#eafaf0 0%,#f4f7f5 40%,#eef6fb 100%);color:var(--text);min-height:100vh;padding:24px 16px 60px}
.shell{max-width:980px;margin:auto}
header.top{display:flex;align-items:center;justify-content:space-between;margin-bottom:24px;flex-wrap:wrap;gap:12px}
.brand{display:flex;align-items:center;gap:14px}
.brand .avatar{font-size:42px;background:var(--primary-light);border-radius:16px;width:64px;height:64px;display:flex;align-items:center;justify-content:center}
.brand h1{font-size:1.7em;font-weight:800;color:var(--primary-dark)}
.brand p{color:var(--muted)}
.lang-switch{display:flex;background:white;border-radius:30px;padding:4px;box-shadow:var(--shadow)}
.lang-switch button{border:none;background:transparent;padding:9px 18px;border-radius:30px;font-weight:600;cursor:pointer}
.lang-switch button.active{background:var(--primary);color:white}
.tabs{display:flex;gap:8px;margin-bottom:22px;background:white;padding:6px;border-radius:16px;box-shadow:var(--shadow);overflow-x:auto}
.tab-btn{flex:1;min-width:140px;padding:13px 10px;border:none;background:transparent;border-radius:12px;font-weight:600;color:var(--muted);cursor:pointer}
.tab-btn.active{background:var(--primary);color:white}
.panel{display:none}
.panel.active{display:block}
.card{background:white;border-radius:var(--radius);box-shadow:var(--shadow);padding:28px;margin-bottom:20px}
.upload-box{border:2.5px dashed #b7e0c6;border-radius:var(--radius);padding:36px 24px;text-align:center;background:var(--primary-light)}
.upload-box h2{margin-bottom:8px}
.upload-box p{color:var(--muted);margin-bottom:18px}
#imageInput{display:none}
.btn{background:linear-gradient(135deg,var(--primary),var(--primary-dark));color:white;padding:14px 32px;border:none;border-radius:14px;font-weight:700;cursor:pointer}
.btn.secondary{background:white;color:var(--primary-dark);border:2px solid var(--primary)}
.btn.block{width:100%;margin-top:18px}
.btn:disabled{opacity:.5;cursor:not-allowed}
#imagePreview{max-width:320px;width:100%;border-radius:16px;margin:20px auto 0}
.loading{display:none;text-align:center;margin:22px 0}
.spinner{border:4px solid #e4f4ea;border-top:4px solid var(--primary);border-radius:50%;width:36px;height:36px;animation:spin .9s linear infinite;margin:auto}
@keyframes spin{to{transform:rotate(360deg)}}
.result{display:none;margin-top:22px}
.result.show{display:block}
.result-card{border-radius:16px;padding:22px;border-left:5px solid var(--primary);background:var(--primary-light)}
.result-card.warning{border-left-color:var(--danger);background:#fdecea}
.result-title{font-weight:700;margin-bottom:10px}
.result-text{white-space:pre-wrap;line-height:1.75}
.farmie-msg{background:white;border-radius:14px;padding:16px;margin-top:16px;border:1.5px solid #dcefe2}
audio{width:100%;margin-top:16px}
.chat-window{height:420px;overflow-y:auto;display:flex;flex-direction:column;gap:12px}
.msg{max-width:78%;padding:12px 16px;border-radius:16px;line-height:1.55}
.msg.bot{align-self:flex-start;background:var(--primary-light)}
.msg.user{align-self:flex-end;background:var(--primary);color:white}
.speak-btn{border:none;background:none;cursor:pointer;margin-top:8px}
.chat-input-row{display:flex;gap:10px;margin-top:16px}
.chat-input-row input{flex:1;padding:14px 16px;border-radius:14px;border:2px solid #e1ede4;font-size:1em}
.icon-btn{width:50px;height:50px;border-radius:14px;border:none;background:var(--primary-light);cursor:pointer;font-size:1.2em}
.icon-btn.send{background:var(--primary);color:white}
.sensor-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:16px;margin-bottom:18px}
.sensor-card{background:var(--primary-light);border-radius:16px;padding:20px;text-align:center}
.sensor-card .icon{font-size:1.8em}
.sensor-card .value{font-size:1.7em;font-weight:800;color:var(--primary-dark)}
.sensor-card .label{color:var(--muted)}
.sensor-card.on{box-shadow:inset 0 0 0 2px var(--primary)}
.sensor-actions{display:flex;gap:12px;flex-wrap:wrap}
.updated-note{color:var(--muted);font-size:.85em;margin-top:12px}
.badge{padding:5px 12px;border-radius:20px;font-size:.8em;font-weight:700}
.badge.live{background:#dbf5e4;color:var(--primary-dark)}
.badge.offline{background:#fdecea;color:#a33a31}
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
<button class="tab-btn active" id="tabDetectBtn" onclick="showTab('detect')">📷 <span id="tabDetectLabel">Detect Disease</span></button>
<button class="tab-btn" id="tabChatBtn" onclick="showTab('chat')">💬 <span id="tabChatLabel">Ask Farmie</span></button>
<button class="tab-btn" id="tabSensorsBtn" onclick="showTab('sensors')">🌡️ <span id="tabSensorsLabel">Farm Sensors</span></button>
</div>
<div class="panel active" id="panel-detect">
<div class="card">
<div class="upload-box">
<h2 id="uploadTitle">📸 Upload Your Crop Image</h2>
<p id="uploadText">Choose a clear photo of the affected leaf</p>
<input type="file" id="imageInput" accept="image/*">
<button class="btn" id="uploadBtn" onclick="document.getElementById('imageInput').click()">📤 Choose Image</button>
</div>
<img id="imagePreview" alt="Preview" style="display:none;">
<button class="btn block" id="detectBtn" style="display:none;" onclick="detectDisease()">🔎 Ask Farmie to Analyze</button>
<div class="loading" id="loading">
<div class="spinner"></div>
<p id="loadingText">Analyzing...</p>
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
<div class="panel" id="panel-chat">
<div class="card">
<div class="chat-window" id="chatWindow"></div>
<div class="chat-input-row">
<button class="icon-btn" id="micBtn" onclick="toggleMic()">🎤</button>
<input type="text" id="chatInput" placeholder="Ask about watering, pests, fertilizer..." onkeydown="if(event.key==='Enter') sendChat();">
<button class="icon-btn send" onclick="sendChat()">➤</button>
</div>
</div>
</div>
<div class="panel" id="panel-sensors">
<div class="card">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;flex-wrap:wrap;gap:10px;">
<h2 id="sensorHeading" style="color:var(--primary-dark);">🌡️ Live Farm Conditions</h2>
<span class="badge offline" id="sensorBadge">OFFLINE</span>
</div>
<div class="sensor-grid" id="sensorGrid"></div>
<div class="sensor-actions">
<button class="btn secondary" id="refreshBtn" onclick="refreshSensors()">🔄 <span id="refreshLabel">Refresh</span></button>
<button class="btn" id="irrigationBtn" onclick="toggleIrrigation()">💧 <span id="irrigationLabel">Turn Irrigation On</span></button>
</div>
<p class="updated-note" id="updatedNote">Waiting for sensor data...</p>
</div>
</div>
</div>
<script>
let lang="english";
let sensorData=null;
let chatStarted=false;
let mediaRecognition=null;
let micActive=false;

const T={
english:{
brandTitle:"Farmie",
brandSubtitle:"Your AI Crop Care Assistant",
tabDetect:"Detect Disease",
tabChat:"Ask Farmie",
tabSensors:"Farm Sensors",
uploadTitle:"📸 Upload Your Crop Image",
uploadText:"Choose a clear photo of the affected leaf",
uploadBtn:"📤 Choose Image",
detectBtn:"🔎 Ask Farmie to Analyze",
loadingText:"Analyzing...",
chatPlaceholder:"Ask about watering, pests, fertilizer...",
sensorHeading:"🌡️ Live Farm Conditions",
refreshLabel:"Refresh",
irrigationOn:"Turn Irrigation Off",
irrigationOff:"Turn Irrigation On",
temp:"Temperature",
humidity:"Humidity",
soil:"Soil Moisture",
irrigation:"Irrigation",
updated:"Last updated: ",
chatWelcome:"Hello! I'm Farmie 🌱 Ask me about watering, fertilizer, pests, or crop disease.",
micNotSupported:"Voice input isn't supported in this browser.",
listening:"Listening...",
selectImage:"Please select an image first.",
online:"LIVE",
offline:"OFFLINE",
pumpError:"Unable to change irrigation."
},
tamil:{
brandTitle:"ஃபார்மி",
brandSubtitle:"உங்கள் AI பயிர் பராமரிப்பு உதவியாளர்",
tabDetect:"நோய் கண்டறிதல்",
tabChat:"ஃபார்மியிடம் கேளுங்கள்",
tabSensors:"பண்ணை சென்சார்கள்",
uploadTitle:"📸 உங்கள் பயிரின் படத்தை பதிவேற்றவும்",
uploadText:"பாதிக்கப்பட்ட இலையின் தெளிவான படத்தை தேர்ந்தெடுக்கவும்",
uploadBtn:"📤 படத்தைத் தேர்ந்தெடுக்கவும்",
detectBtn:"🔎 ஃபார்மியிடம் பகுப்பாய்வு கேளுங்கள்",
loadingText:"பகுப்பாய்வு செய்கிறது...",
chatPlaceholder:"நீர்ப்பாசனம், பூச்சி, உரம் பற்றி கேளுங்கள்...",
sensorHeading:"🌡️ நேரடி பண்ணை நிலைமைகள்",
refreshLabel:"புதுப்பிக்க",
irrigationOn:"பாசனத்தை நிறுத்து",
irrigationOff:"பாசனத்தை இயக்கு",
temp:"வெப்பநிலை",
humidity:"ஈரப்பதம்",
soil:"மண் ஈரப்பதம்",
irrigation:"பாசனம்",
updated:"கடைசியாக புதுப்பிக்கப்பட்டது: ",
chatWelcome:"வணக்கம்! நான் ஃபார்மி 🌱 நீர்ப்பாசனம், உரம், பூச்சிகள் பற்றி கேளுங்கள்.",
micNotSupported:"இந்த உலாவியில் குரல் உள்ளீடு ஆதரிக்கப்படவில்லை.",
listening:"கேட்கிறது...",
selectImage:"முதலில் ஒரு படத்தை தேர்ந்தெடுக்கவும்.",
online:"நேரலை",
offline:"ஆஃப்லைன்",
pumpError:"பாசனத்தை மாற்ற முடியவில்லை."
}};

function setLang(l){
lang=l;
const t=T[lang];
document.getElementById("langEnBtn").classList.toggle("active",l==="english");
document.getElementById("langTaBtn").classList.toggle("active",l==="tamil");
document.getElementById("brandTitle").textContent=t.brandTitle;
document.getElementById("brandSubtitle").textContent=t.brandSubtitle;
document.getElementById("tabDetectLabel").textContent=t.tabDetect;
document.getElementById("tabChatLabel").textContent=t.tabChat;
document.getElementById("tabSensorsLabel").textContent=t.tabSensors;
document.getElementById("uploadTitle").textContent=t.uploadTitle;
document.getElementById("uploadText").textContent=t.uploadText;
document.getElementById("uploadBtn").textContent=t.uploadBtn;
document.getElementById("detectBtn").textContent=t.detectBtn;
document.getElementById("loadingText").textContent=t.loadingText;
document.getElementById("chatInput").placeholder=t.chatPlaceholder;
document.getElementById("sensorHeading").textContent=t.sensorHeading;
document.getElementById("refreshLabel").textContent=t.refreshLabel;
renderSensors();
}

function showTab(name){
document.querySelectorAll(".panel").forEach(p=>p.classList.remove("active"));
document.querySelectorAll(".tab-btn").forEach(b=>b.classList.remove("active"));
document.getElementById("panel-"+name).classList.add("active");
const buttonId="tab"+name.charAt(0).toUpperCase()+name.slice(1)+"Btn";
document.getElementById(buttonId).classList.add("active");
if(name==="sensors") refreshSensors();
if(name==="chat"&&!chatStarted) startChat();
}

document.getElementById("imageInput").addEventListener("change",function(e){
const file=e.target.files[0];
if(!file)return;
const reader=new FileReader();
reader.onload=function(event){
const preview=document.getElementById("imagePreview");
preview.src=event.target.result;
preview.style.display="block";
document.getElementById("detectBtn").style.display="block";
};
reader.readAsDataURL(file);
});

async function detectDisease(){
const file=document.getElementById("imageInput").files[0];
if(!file){
alert(T[lang].selectImage);
return;
}
const loading=document.getElementById("loading");
const result=document.getElementById("result");
const detectBtn=document.getElementById("detectBtn");
loading.style.display="block";
result.classList.remove("show");
detectBtn.disabled=true;
const formData=new FormData();
formData.append("file",file);
formData.append("language",lang);
try{
const response=await fetch("/api/detect",{method:"POST",body:formData});
const data=await response.json();
if(!response.ok)throw new Error(data.error||"Detection failed");
document.getElementById("resultTitle").textContent=data.title;
document.getElementById("resultText").textContent=data.result;
document.getElementById("farmieMsg").textContent=data.farmie_msg;
document.getElementById("resultCard").classList.toggle("warning",!data.healthy);
if(data.audio)document.getElementById("audio").src="data:audio/mp3;base64,"+data.audio;
result.classList.add("show");
}catch(error){
alert("Error: "+error.message);
}finally{
loading.style.display="none";
detectBtn.disabled=false;
}
}

function startChat(){
if(chatStarted)return;
chatStarted=true;
addMessage(T[lang].chatWelcome,"bot",false);
}

function addMessage(text,who,withSpeak=true){
const win=document.getElementById("chatWindow");
const div=document.createElement("div");
div.className="msg "+who;
div.textContent=text;
if(withSpeak){
const button=document.createElement("button");
button.className="speak-btn";
button.textContent="🔊";
button.onclick=()=>speakText(text);
div.appendChild(document.createElement("br"));
div.appendChild(button);
}
win.appendChild(div);
win.scrollTop=win.scrollHeight;
}

async function sendChat(){
const input=document.getElementById("chatInput");
const message=input.value.trim();
if(!message)return;
addMessage(message,"user",false);
input.value="";
try{
const response=await fetch("/api/chat",{
method:"POST",
headers:{"Content-Type":"application/json"},
body:JSON.stringify({message:message,language:lang})
});
const data=await response.json();
if(!response.ok)throw new Error(data.error||"Chat failed");
addMessage(data.reply,"bot");
}catch(error){
addMessage("Error: "+error.message,"bot",false);
}
}

async function speakText(text){
try{
const response=await fetch("/api/tts",{
method:"POST",
headers:{"Content-Type":"application/json"},
body:JSON.stringify({text:text,language:lang})
});
const data=await response.json();
if(data.audio){
const audio=new Audio("data:audio/mp3;base64,"+data.audio);
audio.play();
}
}catch(error){
console.error(error);
}
}

function toggleMic(){
const SpeechRecognition=window.SpeechRecognition||window.webkitSpeechRecognition;
if(!SpeechRecognition){
alert(T[lang].micNotSupported);
return;
}
if(micActive&&mediaRecognition){
mediaRecognition.stop();
return;
}
mediaRecognition=new SpeechRecognition();
mediaRecognition.lang=lang==="tamil"?"ta-IN":"en-US";
mediaRecognition.interimResults=false;
mediaRecognition.maxAlternatives=1;
mediaRecognition.onstart=function(){
micActive=true;
document.getElementById("micBtn").style.background="#ffdada";
document.getElementById("chatInput").placeholder=T[lang].listening;
};
mediaRecognition.onresult=function(event){
const transcript=event.results[0][0].transcript;
document.getElementById("chatInput").value=transcript;
sendChat();
};
mediaRecognition.onerror=function(){
micActive=false;
};
mediaRecognition.onend=function(){
micActive=false;
document.getElementById("micBtn").style.background="";
document.getElementById("chatInput").placeholder=T[lang].chatPlaceholder;
};
mediaRecognition.start();
}

function renderSensors(){
if(!sensorData){
document.getElementById("sensorGrid").innerHTML=`
<div class="sensor-card"><div class="icon">🌡️</div><div class="value">--</div><div class="label">${T[lang].temp}</div></div>
<div class="sensor-card"><div class="icon">💦</div><div class="value">--</div><div class="label">${T[lang].humidity}</div></div>
<div class="sensor-card"><div class="icon">🌱</div><div class="value">--</div><div class="label">${T[lang].soil}</div></div>
<div class="sensor-card"><div class="icon">💧</div><div class="value">--</div><div class="label">${T[lang].irrigation}</div></div>`;
return;
}
const t=T[lang];
const temp=sensorData.temperature_c!==null?sensorData.temperature_c+"°C":"--";
const hum=sensorData.humidity_pct!==null?sensorData.humidity_pct+"%":"--";
const soil=sensorData.soil_status||"UNKNOWN";
const pump=sensorData.irrigation_on?"ON":"OFF";
document.getElementById("sensorGrid").innerHTML=`
<div class="sensor-card"><div class="icon">🌡️</div><div class="value">${temp}</div><div class="label">${t.temp}</div></div>
<div class="sensor-card"><div class="icon">💦</div><div class="value">${hum}</div><div class="label">${t.humidity}</div></div>
<div class="sensor-card"><div class="icon">🌱</div><div class="value">${soil}</div><div class="label">${t.soil}</div></div>
<div class="sensor-card ${sensorData.irrigation_on?"on":""}"><div class="icon">💧</div><div class="value">${pump}</div><div class="label">${t.irrigation}</div></div>`;
document.getElementById("irrigationLabel").textContent=sensorData.irrigation_on?t.irrigationOn:t.irrigationOff;
const badge=document.getElementById("sensorBadge");
if(sensorData.device_online){
badge.textContent=t.online;
badge.className="badge live";
}else{
badge.textContent=t.offline;
badge.className="badge offline";
}
if(sensorData.last_updated>0){
const date=new Date(sensorData.last_updated*1000);
document.getElementById("updatedNote").textContent=t.updated+date.toLocaleTimeString();
}
}

async function refreshSensors(){
const btn=document.getElementById("refreshBtn");
btn.disabled=true;
try{
const response=await fetch("/api/sensors");
if(!response.ok)throw new Error("Sensor server error");
sensorData=await response.json();
renderSensors();
}catch(error){
console.error("Sensor error:",error);
}finally{
btn.disabled=false;
}
}

async function toggleIrrigation(){
const button=document.getElementById("irrigationBtn");
button.disabled=true;
try{
const current=sensorData&&sensorData.irrigation_on?true:false;
const newState=!current;
const response=await fetch("/api/irrigation",{
method:"POST",
headers:{"Content-Type":"application/json"},
body:JSON.stringify({irrigation_on:newState})
});
const data=await response.json();
if(!response.ok)throw new Error(data.error||"Irrigation command failed");
sensorData=data.state;
renderSensors();
}catch(error){
console.error(error);
alert(T[lang].pumpError+" "+error.message);
}finally{
button.disabled=false;
}
}

renderSensors();
refreshSensors();

setInterval(function(){
if(document.getElementById("panel-sensors").classList.contains("active")){
refreshSensors();
}
},15000);
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
        print("Detect Error:", e)
        return jsonify({"error": f"Detection failed: {e}"}), 500

def _run_detect():
    if "file" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["file"]
    language = request.form.get("language", "english")
    lang_code = "ta" if language == "tamil" else "en"

    image = Image.open(file).convert("RGB").resize((224, 224))

    if MODEL_READY:
        input_data = np.array(image, dtype=np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)

        interpreter.set_tensor(
            input_details[0]["index"],
            input_data
        )
        interpreter.invoke()

        output = interpreter.get_tensor(
            output_details[0]["index"]
        )

        predicted_class = labels[int(np.argmax(output[0]))]
        confidence = float(np.max(output[0])) * 100
    else:
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
            farmie_msg = "🌱 ஃபார்மி: உங்கள் பயிர் ஆரோக்கியமாக இருக்கிறது!"
        else:
            title = "✅ Great! Your Crop is Healthy!"
            result = (
                f"🌿 Status: HEALTHY\n"
                f"📊 Confidence: {confidence:.1f}%\n\n"
                f"Your crop appears to be healthy."
            )
            farmie_msg = "🌱 Farmie: Your crop looks healthy!"
    else:
        remedy = get_remedy(predicted_class, lang_code)

        if language == "tamil":
            title = f"⚠️ பயிர் நோய்: {predicted_class}"
            result = (
                f"🌿 நோய்: {predicted_class}\n"
                f"📊 நம்பிக்கை: {confidence:.1f}%\n\n"
                f"💊 தீர்வு:\n{remedy}"
            )
            farmie_msg = f"🌱 ஃபார்மி: {predicted_class} கண்டறியப்பட்டுள்ளது."
        else:
            title = f"⚠️ Crop Disease Detected: {predicted_class}"
            result = (
                f"🌿 Disease: {predicted_class}\n"
                f"📊 Confidence: {confidence:.1f}%\n\n"
                f"💊 Recommended Remedy:\n{remedy}"
            )
            farmie_msg = f"🌱 Farmie: {predicted_class} detected."

    audio_base64 = _tts_base64(farmie_msg, lang_code)

    return jsonify({
        "title": title,
        "result": result,
        "farmie_msg": farmie_msg,
        "audio": audio_base64,
        "healthy": healthy,
        "demo_mode": not MODEL_READY
    })

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "")
    language = data.get("language", "english")

    reply = answer_query(
        message,
        "en" if language == "english" else "ta"
    )

    return jsonify({"reply": reply})

@app.route("/api/tts", methods=["POST"])
def tts():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    language = data.get("language", "english")
    lang_code = "ta" if language == "tamil" else "en"

    audio_base64 = _tts_base64(text, lang_code)

    return jsonify({"audio": audio_base64})

@app.route("/api/esp32/data", methods=["POST"])
def esp32_data():
    data = request.get_json(silent=True) or {}

    try:
        soil = str(data.get("soil_status", "UNKNOWN")).upper()
        temperature = float(data.get("temperature_c"))
        humidity = float(data.get("humidity_pct"))
        pump = str(data.get("pump", "OFF")).upper()

        if soil not in ["DRY", "WET"]:
            return jsonify({"error": "Invalid soil status"}), 400

        if pump not in ["ON", "OFF"]:
            return jsonify({"error": "Invalid pump status"}), 400

        sensor_state["soil_status"] = soil
        sensor_state["temperature_c"] = temperature
        sensor_state["humidity_pct"] = humidity
        sensor_state["irrigation_on"] = pump == "ON"
        sensor_state["last_updated"] = time.time()

        return jsonify({
            "success": True,
            "message": "Sensor data received",
            "irrigation_command": irrigation_command
        })
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid sensor data"}), 400

@app.route("/api/sensors", methods=["GET"])
def sensors():
    state = dict(sensor_state)

    if state["last_updated"] == 0:
        state["device_online"] = False
    else:
        state["device_online"] = time.time() - state["last_updated"] < 15

    return jsonify(state)

@app.route("/api/irrigation", methods=["POST"])
def irrigation():
    global irrigation_command
    global command_updated

    data = request.get_json(silent=True) or {}

    if "irrigation_on" not in data:
        return jsonify({"error": "Missing irrigation_on"}), 400

    irrigation_command = bool(data["irrigation_on"])
    command_updated = time.time()

    sensor_state["irrigation_on"] = irrigation_command

    return jsonify({
        "success": True,
        "message": "Irrigation command updated",
        "state": sensor_state
    })

@app.route("/api/esp32/command", methods=["GET"])
def esp32_command():
    return jsonify({
        "irrigation_on": irrigation_command,
        "updated": command_updated
    })

def _tts_base64(text, lang_code, timeout=8):
    if not text:
        return None

    def _generate():
        tts_obj = gTTS(text=text, lang=lang_code)
        buf = io.BytesIO()
        tts_obj.write_to_fp(buf)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_generate)
            return future.result(timeout=timeout)
    except Exception as e:
        print("TTS Error:", e)
        return None

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
