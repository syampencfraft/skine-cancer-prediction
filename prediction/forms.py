from django import forms
from .models import Scan

class SkinCancerPredictionForm(forms.ModelForm):
    class Meta:
        model = Scan
        fields = ['image']
        widgets = {
            'image': forms.FileInput(attrs={'class': 'form-control', 'accept': 'image/*'})
        }
