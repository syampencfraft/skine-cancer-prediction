from django.shortcuts import render, redirect
from .forms import SkinCancerPredictionForm
from .models import Scan
from .utils import load_model_and_predict

from django.contrib.auth.decorators import login_required

def home(request):
    return render(request, 'prediction/index.html')

@login_required
def predict(request):
    if request.method == 'POST':
        form = SkinCancerPredictionForm(request.POST, request.FILES)
        if form.is_valid():
            scan = form.save(commit=False)
            scan.user = request.user
            scan.save()
            
            # Perform prediction
            label, confidence, risk_level, recommendation = load_model_and_predict(scan.image.path)
            
            # Update scan with result
            scan.result = label
            scan.confidence = confidence
            scan.risk_level = risk_level
            scan.recommendation = recommendation
            scan.save()
            
            return render(request, 'prediction/result.html', {'scan': scan})
    else:
        form = SkinCancerPredictionForm()
    
    return render(request, 'prediction/predict.html', {'form': form})

@login_required
def history(request):
    scans = Scan.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'prediction/history.html', {'scans': scans})
