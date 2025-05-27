# In detection/urls.py
from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('', views.home, name='home'),
    path('register/', views.register, name='register'),
    path('login/', auth_views.LoginView.as_view(template_name='login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(template_name='logout.html'), name='logout'),
    path('live-detection/', views.live_detection, name='live_detection'),
    path('video-feed/', views.live_feed, name='video_feed'),
    path('detection-history/', views.detection_history, name='detection_history'),
    path('upload/', views.image_upload, name='image_upload'),
    path('upload/<int:pk>/', views.image_detail, name='image_detail'),
    path('upload-history/', views.upload_history, name='upload_history'),
]
