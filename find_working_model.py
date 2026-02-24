import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

with open("available_models.txt", "r") as f:
    lines = f.readlines()

models_to_test = [line.strip().replace("- ", "") for line in lines if line.startswith("- ")]

print(f"Testing {len(models_to_test)} models...")

for model_name in models_to_test:
    try:
        print(f"Testing {model_name}...", end=" ", flush=True)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Ping", generation_config={"max_output_tokens": 5})
        print("SUCCESS")
        with open("working_model.txt", "w") as f:
            f.write(model_name)
        break
    except Exception as e:
        print(f"FAILED: {str(e)[:50]}")
