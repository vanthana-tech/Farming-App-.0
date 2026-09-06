"""
remedies.py
Provides disease/pest remedy lookup and sensor-based field condition
evaluation for the Smart Farming Assistant demo app.

Designed to work with standard PlantVillage-style labels, e.g.:
    "Tomato___Early_blight", "Potato___Late_blight", "Corn___healthy"

Also supports plain labels like "Early_blight" or "healthy" if your
dataset/model uses simpler class names.
"""

STATUS_COLORS = {
    "healthy": "✅ ",
    "disease": "🔴 ",
    "warning": "⚠️ ",
    "info": "ℹ️ ",
}

# ---------- Specific overrides for common PlantVillage diseases ----------
# Add more entries here as you find them in your actual labels.json
SPECIFIC_REMEDIES = {
    "early_blight": {
        "en": ("Early Blight detected", "Remove infected leaves. Apply "
               "copper-based or chlorothalonil fungicide. Avoid overhead "
               "watering; water at the base of the plant."),
        "ta": ("ஆரம்ப கருகல் நோய் கண்டறியப்பட்டது", "பாதிக்கப்பட்ட இலைகளை "
               "அகற்றவும். காப்பர் அடிப்படையிலான பூஞ்சைக் கொல்லியைத் "
               "தெளிக்கவும். மேலிருந்து நீர் பாய்ச்சுவதைத் தவிர்க்கவும்."),
    },
    "late_blight": {
        "en": ("Late Blight detected", "This spreads fast — remove and "
               "destroy infected plants immediately. Apply fungicide "
               "(mancozeb or metalaxyl) and improve field drainage."),
        "ta": ("பிந்தைய கருகல் நோய் கண்டறியப்பட்டது", "இது வேகமாக பரவும் — "
               "பாதிக்கப்பட்ட செடிகளை உடனடியாக அகற்றவும். பூஞ்சைக் "
               "கொல்லியைத் தெளிக்கவும்."),
    },
    "leaf_mold": {
        "en": ("Leaf Mold detected", "Improve ventilation, reduce humidity "
               "around plants, and apply a suitable fungicide."),
        "ta": ("இலை பூஞ்சை கண்டறியப்பட்டது", "காற்றோட்டத்தை மேம்படுத்தவும், "
               "ஈரப்பதத்தை குறைக்கவும், பூஞ்சைக் கொல்லியைத் தெளிக்கவும்."),
    },
    "bacterial_spot": {
        "en": ("Bacterial Spot detected", "Remove affected leaves, avoid "
               "working in wet fields, and apply copper-based bactericide."),
        "ta": ("பாக்டீரியா புள்ளி நோய் கண்டறியப்பட்டது", "பாதிக்கப்பட்ட "
               "இலைகளை அகற்றவும். காப்பர் அடிப்படையிலான மருந்தைப் "
               "பயன்படுத்தவும்."),
    },
    "mosaic_virus": {
        "en": ("Mosaic Virus detected", "No cure — remove and destroy "
               "infected plants to stop spread. Control aphids/whiteflies, "
               "which spread the virus."),
        "ta": ("மொசைக் வைரஸ் கண்டறியப்பட்டது", "குணப்படுத்த முடியாது — "
               "பாதிக்கப்பட்ட செடிகளை அகற்றவும். பரப்பும் பூச்சிகளை "
               "கட்டுப்படுத்தவும்."),
    },
    "powdery_mildew": {
        "en": ("Powdery Mildew detected", "Apply sulfur-based fungicide, "
               "improve air circulation, avoid overcrowding plants."),
        "ta": ("பொடி பூஞ்சை நோய் கண்டறியப்பட்டது", "சல்பர் அடிப்படையிலான "
               "பூஞ்சைக் கொல்லியைத் தெளிக்கவும். காற்றோட்டத்தை "
               "மேம்படுத்தவும்."),
    },
    "leaf_rust": {
        "en": ("Leaf Rust detected", "Apply appropriate fungicide (e.g. "
               "propiconazole) and remove heavily infected leaves."),
        "ta": ("இலை துரு நோய் கண்டறியப்பட்டது", "பொருத்தமான பூஞ்சைக் "
               "கொல்லியைத் தெளிக்கவும். கடுமையாக பாதிக்கப்பட்ட இலைகளை "
               "அகற்றவும்."),
    },
    "spider_mites": {
        "en": ("Spider Mite infestation detected", "Spray with neem oil or "
               "insecticidal soap. Increase humidity, as mites thrive in "
               "dry conditions."),
        "ta": ("சிலந்தி பூச்சி தாக்குதல் கண்டறியப்பட்டது", "வேப்பெண்ணெய் "
               "தெளிக்கவும். ஈரப்பதத்தை அதிகரிக்கவும்."),
    },
    "target_spot": {
        "en": ("Target Spot detected", "Remove infected debris, rotate "
               "crops, and apply a broad-spectrum fungicide."),
        "ta": ("இலக்கு புள்ளி நோய் கண்டறியப்பட்டது", "பாதிக்கப்பட்ட "
               "பொருட்களை அகற்றவும், பயிர் சுழற்சி செய்யவும்."),
    },
}

# Generic fallback templates, matched by keyword found in the label
KEYWORD_FALLBACKS = {
    "blight": SPECIFIC_REMEDIES["early_blight"],
    "mold": SPECIFIC_REMEDIES["leaf_mold"],
    "spot": SPECIFIC_REMEDIES["bacterial_spot"],
    "mosaic": SPECIFIC_REMEDIES["mosaic_virus"],
    "virus": SPECIFIC_REMEDIES["mosaic_virus"],
    "mildew": SPECIFIC_REMEDIES["powdery_mildew"],
    "rust": SPECIFIC_REMEDIES["leaf_rust"],
    "mite": SPECIFIC_REMEDIES["spider_mites"],
    "scab": SPECIFIC_REMEDIES["bacterial_spot"],
}

HEALTHY_MSG = {
    "en": ("Plant looks healthy", ""),
    "ta": ("செடி ஆரோக்கியமாக உள்ளது", ""),
}

UNKNOWN_MSG = {
    "en": ("Issue detected", "Could not match a specific remedy for this "
           "class. Please consult a local agriculture officer or add this "
           "label to remedies.py."),
    "ta": ("பிரச்சனை கண்டறியப்பட்டது", "குறிப்பிட்ட தீர்வு கிடைக்கவில்லை. "
           "உள்ளூர் வேளாண் அலுவலரை அணுகவும்."),
}

# ---------- Sensor-based field conditions ----------
SENSOR_MSG = {
    "drought_risk": {
        "en": ("Drought risk — soil moisture is low",
               "Irrigate soon. Consider mulching to retain soil moisture."),
        "ta": ("வறட்சி ஆபத்து — மண் ஈரப்பதம் குறைவு",
               "விரைவில் நீர்ப்பாசனம் செய்யவும். மல்ச்சிங் பயன்படுத்தவும்."),
    },
    "overwatering_risk": {
        "en": ("Overwatering risk — soil moisture is very high",
               "Reduce irrigation and improve drainage to avoid root rot."),
        "ta": ("அதிக நீர் ஆபத்து — மண் ஈரப்பதம் மிக அதிகம்",
               "நீர்ப்பாசனத்தை குறைக்கவும். வடிகால் மேம்படுத்தவும்."),
    },
    "heat_stress": {
        "en": ("Heat stress risk — temperature is high",
               "Irrigate during cooler hours (early morning/evening) and "
               "consider shade netting."),
        "ta": ("வெப்ப அழுத்த ஆபத்து — வெப்பநிலை அதிகம்",
               "குளிர்ந்த நேரங்களில் நீர்ப்பாசனம் செய்யவும்."),
    },
    "normal": {
        "en": ("Field conditions are normal", ""),
        "ta": ("வயல் நிலைமைகள் இயல்பாக உள்ளன", ""),
    },
}


def evaluate_sensors(soil_moisture, temperature):
    """Return a condition key based on simple threshold rules.
    Tune these thresholds for your target crop."""
    if soil_moisture < 30:
        return "drought_risk"
    if soil_moisture > 85:
        return "overwatering_risk"
    if temperature > 38:
        return "heat_stress"
    return "normal"


def _status_for(key):
    if key == "normal" or key == "healthy":
        return "healthy"
    if key in ("drought_risk", "overwatering_risk", "heat_stress"):
        return "warning"
    return "disease"


def get_remedy(class_label, lang_code="en"):
    """
    Look up remedy info for either:
      - a model prediction label (e.g. 'Tomato___Early_blight')
      - a sensor condition key (e.g. 'drought_risk')
    Returns dict: {status, message, remedy}
    """
    key = class_label.lower()

    # Sensor condition keys
    if key in SENSOR_MSG:
        title, remedy = SENSOR_MSG[key][lang_code]
        return {"status": _status_for(key), "message": title, "remedy": remedy}

    # Healthy classes (e.g. "Tomato___healthy", "healthy")
    if "healthy" in key:
        title, remedy = HEALTHY_MSG[lang_code]
        return {"status": "healthy", "message": title, "remedy": remedy}

    # Normalize: collapse any run of underscores/spaces into a single "_"
    import re
    normalized = re.sub(r"[_\s]+", "_", key).strip("_")

    # Specific match first (checked as substring so "tomato_early_blight"
    # matches "early_blight" before the generic "blight" keyword fires)
    for disease_key, data in SPECIFIC_REMEDIES.items():
        if disease_key in normalized:
            title, remedy = data[lang_code]
            return {"status": "disease", "message": title, "remedy": remedy}

    # Keyword fallback match (broader, less specific)
    for keyword, data in KEYWORD_FALLBACKS.items():
        if keyword in normalized:
            title, remedy = data[lang_code]
            return {"status": "disease", "message": title, "remedy": remedy}

    # Unknown — generic message
    title, remedy = UNKNOWN_MSG[lang_code]
    return {"status": "warning", "message": title, "remedy": remedy}
