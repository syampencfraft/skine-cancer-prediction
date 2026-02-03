
import os
import django
import sys
from PIL import Image
import numpy as np

# Setup Django environment
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'skin_cancer_detection.settings')
django.setup()

from prediction.utils import load_model_and_predict

# Create a dummy image
img_path = 'test_image.jpg'
img = Image.new('RGB', (224, 224), color = 'red')
img.save(img_path)

print(f"Testing prediction on {img_path}...")
try:
    label, confidence = load_model_and_predict(img_path)
    print(f"Success! Prediction: {label}, Confidence: {confidence}%")
except Exception as e:
    print(f"FAILED: {e}")
finally:
    if os.path.exists(img_path):
        os.remove(img_path)
