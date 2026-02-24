import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

try:
    # Explicitly using 'models/' prefix
    model = genai.GenerativeModel('models/gemini-1.5-flash')
    response = model.generate_content("Hello, respond with 'Model is working'")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
