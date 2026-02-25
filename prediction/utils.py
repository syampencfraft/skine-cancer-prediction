import os
import warnings
# Suppress the deprecation warning while we still use google-generativeai
warnings.filterwarnings("ignore", message=".*google.genai.*")
import google.generativeai as genai
from django.conf import settings
from PIL import Image
import json

# Setup Gemini
API_KEY = os.getenv("GEMINI_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

def load_model_and_predict(image_path):
    """
    Predict skin disease using Gemini 1.5 Pro for maximum accuracy.
    Provides diagnosis, confidence, risk level, and medical recommendations.
    """
    
    if not API_KEY or API_KEY == "your_api_key_here":
         return "API Key Not Configured", 0.0, "High", "Please contact administrator."

    try:
        # Use gemini-2.5-flash which is confirmed to work for this API key
        model = genai.GenerativeModel('models/gemini-2.5-flash')
        
        # Load image
        img = Image.open(image_path)
        
        # Rigorous Medical Analysis Prompt
        prompt = """
        You are a highly specialised dermatological AI assistant. Your task is to analyze the provided dermoscopic image of a skin lesion with extreme precision.
        
        Step 1: Determine if the image is a valid, clear dermoscopic or clinical close-up of a human skin lesion.
        If it is NOT a skin lesion (e.g., an object, an animal, a landscape, or a low-quality/blurry image), respond ONLY with this JSON:
        {
            "is_skin_disease": false,
            "message": "this is not a skin deisies image"
        }
        
        Step 2: If it IS a skin lesion, perform a comprehensive analysis:
        1. Identify the most specific medical diagnosis (e.g., 'Basal Cell Carcinoma', 'Seborrheic Keratosis', 'Melanocytic Naevus').
        2. Assign a confidence score (0-100) reflecting your scientific certainty.
        3. Determine the Risk Level (Low, Moderate, High).
        4. Provide brief, actionable medical recommendations (e.g., 'Monitor for changes', 'Consult a dermatologist for a biopsy').
        
        Respond ONLY with this exact JSON format:
        {
            "is_skin_disease": true,
            "diagnosis": "Name of Disease",
            "confidence": 98.5,
            "risk_level": "Low/Moderate/High",
            "recommendation": "Specific medical advice."
        }
        """

        response = model.generate_content([prompt, img])
        
        # Clean response text
        cleaned_text = response.text.strip().replace("```json", "").replace("```", "")
        data = json.loads(cleaned_text)
        
        if not data.get("is_skin_disease", False):
            return data.get("message", "this is not a skin deisies image"), 0.0, "N/A", "Please upload a clear image of a skin lesion."
            
        label = data.get("diagnosis", "Unknown Condition")
        confidence = data.get("confidence", 0.0)
        risk_level = data.get("risk_level", "Unknown")
        recommendation = data.get("recommendation", "Please consult a professional.")
        
        return label, confidence, risk_level, recommendation

    except Exception as e:
        print(f"Gemini API Error: {e}")
        return f"Error: {str(e)[:40]}", 0.0, "Unknown", "An unexpected error occurred during analysis."
