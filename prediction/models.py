from django.db import models

class Scan(models.Model):
    image = models.ImageField(upload_to='scans/')
    result = models.CharField(max_length=255, blank=True)
    confidence = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Scan {self.id} - {self.result}"
