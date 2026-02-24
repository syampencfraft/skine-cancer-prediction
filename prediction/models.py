from django.db import models
from django.contrib.auth.models import User

class Scan(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='scans', null=True, blank=True)
    image = models.ImageField(upload_to='scans/')
    result = models.CharField(max_length=255, blank=True)
    confidence = models.FloatField(null=True, blank=True)
    risk_level = models.CharField(max_length=50, blank=True)
    recommendation = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Scan {self.id} - {self.result}"
