# In detection/models.py
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import os


class Detection(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(default=timezone.now)
    image = models.ImageField(upload_to='detections/%Y/%m/%d/')
    confidence = models.FloatField(default=0.0)
    
    def __str__(self):
        return f"Detection by {self.user.username} at {self.timestamp}"

class UploadedImage(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='uploads/%Y/%m/%d/')
    uploaded_at = models.DateTimeField(default=timezone.now)
    
    # Detection results
    has_weapon = models.BooleanField(default=False)
    confidence = models.FloatField(default=0.0)
    
    # Metadata fields
    original_filename = models.CharField(max_length=255, blank=True)
    file_size = models.IntegerField(default=0)  # In bytes
    image_width = models.IntegerField(default=0)
    image_height = models.IntegerField(default=0)
    
    # EXIF data (if available)
    date_taken = models.DateTimeField(null=True, blank=True)
    camera_make = models.CharField(max_length=100, blank=True)
    camera_model = models.CharField(max_length=100, blank=True)
    location_latitude = models.FloatField(null=True, blank=True)
    location_longitude = models.FloatField(null=True, blank=True)
    
    def __str__(self):
        return f"Image uploaded by {self.user.username} on {self.uploaded_at}"
    
    def filename(self):
        return os.path.basename(self.image.name)
