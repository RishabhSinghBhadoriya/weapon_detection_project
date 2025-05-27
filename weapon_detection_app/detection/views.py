from pathlib import Path
import uuid
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .forms import UserRegistrationForm, ImageUploadForm
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators import gzip
import cv2
import threading
from ultralytics import YOLO
import os
from datetime import datetime
from .models import Detection, UploadedImage
from django.conf import settings
import time
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import datetime
import json

# Global variables
model = None
output_frame = None
lock = threading.Lock()
last_detection_time = 0
detection_cooldown = 2  # seconds between saved detections

# Store active camera handlers per user
active_cameras = {}
camera_lock = threading.Lock()

def load_model():
    global model
    try:
        # Get the directory of the current script (e.g., weapon_detection_app)
        current_script_dir = Path(__file__).resolve().parent

        # Go up one level from current_script_dir to reach WEAPON_DETECTION_PROJECT
        project_root_dir = current_script_dir.parent

        # Construct the full path to best.pt
        model_path = project_root_dir / 'best.pt'

        print(f"Attempting to load model from: {model_path}")

        if os.path.exists(model_path):
            model = YOLO(model_path) # Uncomment this line when you have YOLO installed
            print("✅ YOLO model loaded successfully")
        else:
            print(f"❌ Model file not found at: {model_path}")
            model = None
    except Exception as e:
        print(f"Error loading model: {e}")

# Call this when Django starts
load_model()

@login_required
def home(request):
    return render(request, 'home.html')

def register(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'Account created for {username}! You can now log in.')
            return redirect('login')
    else:
        form = UserRegistrationForm()
    return render(request, 'register.html', {'form': form})

def detect_weapons(frame):
    global model, last_detection_time

    if model is None:
        print("DEBUG (Django): Model is None in detect_weapons. Cannot perform detection.")
        return frame, False, 0, []

    # Check the type and properties of the frame *before* passing to model
    print(f"DEBUG (Django): Frame type before model inference: {type(frame)}")
    if isinstance(frame, (cv2.typing.MatLike,)): # cv2.MatLike is usually numpy.ndarray
        print(f"DEBUG (Django): Frame shape: {frame.shape}, dtype: {frame.dtype}")
    else:
        print(f"WARNING (Django): Frame is not an OpenCV/Numpy array. Type: {type(frame)}")
        # If it's a PIL Image, you might need to convert it to OpenCV format first
        # For example: frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        # However, model() usually handles PIL images too.

    try:
        # Run inference
        results = model(frame) # This is the crucial line

        annotated_frame = frame.copy() # Start with a copy
        detection_found = False
        max_conf = 0
        detections_info = []

        if results and len(results) > 0:
            print(f"DEBUG (Django): Model returned {len(results)} result objects.")
            # YOLO's plot() method expects BGR, which OpenCV provides from webcam
            annotated_frame = results[0].plot() # This should draw the boxes
            print("DEBUG (Django): results[0].plot() executed.")

            for r in results:
                if len(r.boxes) > 0:
                    for box in r.boxes:
                        conf = box.conf.item()
                        if conf > 0.5:  # Confidence threshold
                            detection_found = True
                            max_conf = max(max_conf, conf)
                            class_id = int(box.cls.item())
                            class_name = model.names.get(class_id, f'Class_{class_id}')
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            detections_info.append({
                                'class': class_name, 'confidence': conf, 'bbox': [x1, y1, x2, y2]
                            })
                            print(f"DEBUG (Django): Detected weapon: {class_name} with confidence {conf:.2f}")
                else:
                    print("DEBUG (Django): No boxes in this result object.")
        else:
            print("DEBUG (Django): No results or empty results from model inference.")

        return annotated_frame, detection_found, max_conf, detections_info
    except Exception as e:
        print(f"ERROR (Django) in detect_weapons: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for more info
        return frame, False, 0, []

class CameraHandler:
    def __init__(self, user_id):
        self.camera = None
        self.user_id = user_id
        self.user = None
        self.is_active = False
        self.last_detection_info = None
    
    def initialize_camera(self, user):
        self.user = user
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            print("Error: Could not open webcam")
            return False
        
        # Set camera properties for better performance
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        self.is_active = True
        return True
    
    def generate_frames(self):
        global last_detection_time
        
        if not self.camera or not self.camera.isOpened():
            return
        
        try:
            while self.is_active:
                success, frame = self.camera.read()
                if not success:
                    print("Failed to read frame from camera")
                    break
                
                # Detect weapons in frame
                frame, detection_found, confidence, detections_info = detect_weapons(frame)
                
                # Store detection info for status endpoint
                if detection_found:
                    self.last_detection_info = {
                        'detected': True,
                        'confidence': confidence,
                        'detections': detections_info,
                        'timestamp': time.time()
                    }
                else:
                    self.last_detection_info = {
                        'detected': False,
                        'timestamp': time.time()
                    }
                
                # If weapon detected and cooldown period passed, save the frame
                current_time = time.time()
                if detection_found and (current_time - last_detection_time) > detection_cooldown and self.user:
                    try:
                        file_path = save_frame_locally(frame, self.user)
                        
                        if file_path:
                            # Create database entry
                            detection = Detection(
                                user=self.user,
                                image=file_path,
                                confidence=confidence
                            )
                            detection.save()
                            
                            last_detection_time = current_time
                            print(f"Detection saved: {file_path} with confidence {confidence:.2f}")
                    except Exception as e:
                        print(f"Error saving detection: {e}")
                
                # Convert to JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    
                    # Yield the frame in byte format
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.033)  # ~30 FPS
                
        except Exception as e:
            print(f"Error in frame generation: {e}")
        finally:
            self.cleanup()
    
    def stop(self):
        """Stop the camera stream"""
        self.is_active = False
    
    def cleanup(self):
        """Clean up camera resources"""
        self.is_active = False
        if self.camera:
            self.camera.release()
            self.camera = None
        print(f"Camera cleaned up for user {self.user_id}")

def get_or_create_camera_handler(user):
    """Get existing camera handler or create new one"""
    global active_cameras, camera_lock
    
    with camera_lock:
        user_id = user.id
        
        # Clean up any existing handler for this user
        if user_id in active_cameras:
            active_cameras[user_id].cleanup()
            del active_cameras[user_id]
        
        # Create new handler
        handler = CameraHandler(user_id)
        if handler.initialize_camera(user):
            active_cameras[user_id] = handler
            return handler
        else:
            return None

@gzip.gzip_page
@login_required
def live_feed(request):
    """Stream live video feed with weapon detection"""
    handler = get_or_create_camera_handler(request.user)
    
    if not handler:
        return JsonResponse({'error': 'Could not initialize camera'}, status=500)
    
    return StreamingHttpResponse(
        handler.generate_frames(), 
        content_type='multipart/x-mixed-replace; boundary=frame'
    )

@login_required
def live_detection(request):
    """Live detection page"""
    return render(request, 'live_detection.html')

@login_required
def detection_status(request):
    """Get current detection status"""
    global active_cameras, camera_lock
    
    with camera_lock:
        user_id = request.user.id
        if user_id in active_cameras:
            handler = active_cameras[user_id]
            if handler.last_detection_info:
                return JsonResponse({
                    'status': 'success',
                    'detection_info': handler.last_detection_info
                })
    
    return JsonResponse({
        'status': 'no_data',
        'detection_info': {'detected': False, 'timestamp': time.time()}
    })

@login_required
def stop_camera(request):
    """Stop the camera for the current user"""
    global active_cameras, camera_lock
    
    with camera_lock:
        user_id = request.user.id
        if user_id in active_cameras:
            active_cameras[user_id].cleanup()
            del active_cameras[user_id]
            return JsonResponse({'status': 'success', 'message': 'Camera stopped'})
    
    return JsonResponse({'status': 'info', 'message': 'No active camera found'})

@login_required
def camera_status(request):
    """Check if camera is active for current user"""
    global active_cameras, camera_lock
    
    with camera_lock:
        user_id = request.user.id
        is_active = user_id in active_cameras and active_cameras[user_id].is_active
    
    return JsonResponse({
        'status': 'success',
        'camera_active': is_active
    })

@login_required
def detection_history(request):
    """Show detection history for the current user"""
    detections = Detection.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'detection_history.html', {'detections': detections})

@login_required
def save_detection(request):
    if request.method == 'POST' and request.FILES.get('image'):
        detection = Detection(
            user=request.user,
            image=request.FILES['image'],
            confidence=float(request.POST.get('confidence', 0))
        )
        detection.save()
        return JsonResponse({'status': 'success'})
    return JsonResponse({'status': 'error'})

def save_frame_locally(frame, user):
    """Save a frame locally and return the file path"""
    try:
        # Generate a unique filename
        filename = f"detection_{uuid.uuid4()}.jpg"
        
        # Use media root directly since you don't have detections directory
        file_path = filename
        full_path = os.path.join(settings.MEDIA_ROOT, file_path)
        
        # Ensure media root exists
        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
        
        # Save the image
        success = cv2.imwrite(full_path, frame)
        if not success:
            raise Exception("Failed to write image file")
        
        return file_path
    except Exception as e:
        print(f"Error saving frame: {e}")
        return None

def extract_exif_data(image_path):
    """Extract EXIF data from an image"""
    try:
        image = Image.open(image_path)
        exif_data = {}
        
        if hasattr(image, '_getexif') and image._getexif():
            exif_info = image._getexif()
            for tag, value in exif_info.items():
                decoded = TAGS.get(tag, tag)
                exif_data[decoded] = value
                    
        # Extract GPS data if available
        gps_info = {}
        if 'GPSInfo' in exif_data:
            for key in exif_data['GPSInfo'].keys():
                decoded = GPSTAGS.get(key, key)
                gps_info[decoded] = exif_data['GPSInfo'][key]
        
        # Process metadata
        metadata = {
            'width': image.width,
            'height': image.height,
            'camera_make': exif_data.get('Make', ''),
            'camera_model': exif_data.get('Model', ''),
            'date_taken': None,
            'latitude': None,
            'longitude': None
        }
        
        # Extract date taken
        if 'DateTimeOriginal' in exif_data:
            date_str = exif_data['DateTimeOriginal']
            try:
                date_taken = datetime.datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')
                metadata['date_taken'] = date_taken
            except ValueError:
                pass
        
        # Extract GPS coordinates if available
        if gps_info:
            try:
                lat_ref = gps_info.get('GPSLatitudeRef', 'N')
                lat = gps_info.get('GPSLatitude')
                lon_ref = gps_info.get('GPSLongitudeRef', 'E')
                lon = gps_info.get('GPSLongitude')
                
                if lat and lon and len(lat) >= 3 and len(lon) >= 3:
                    # Convert coordinates to decimal format
                    lat_value = float(lat[0]) + float(lat[1])/60 + float(lat[2])/3600
                    lon_value = float(lon[0]) + float(lon[1])/60 + float(lon[2])/3600
                    
                    if lat_ref == 'S':
                        lat_value = -lat_value
                    if lon_ref == 'W':
                        lon_value = -lon_value
                        
                    metadata['latitude'] = lat_value
                    metadata['longitude'] = lon_value
            except (ValueError, TypeError, IndexError):
                # GPS data extraction can be complex and error-prone
                pass
                
        return metadata
    except Exception as e:
        print(f"Error extracting EXIF data: {e}")
        return {
            'width': 0,
            'height': 0,
            'camera_make': '',
            'camera_model': '',
            'date_taken': None,
            'latitude': None,
            'longitude': None
        }

def process_uploaded_image(image_file, user):
    """Process an uploaded image file"""
    try:
        # Create initial record
        upload = UploadedImage(
            user=user,
            image=image_file,
            original_filename=image_file.name,
            file_size=image_file.size
        )
        upload.save()
        
        # Get full path to saved image
        image_path = upload.image.path
        
        # Extract metadata
        metadata = extract_exif_data(image_path)
        
        # Update record with metadata
        upload.image_width = metadata['width']
        upload.image_height = metadata['height']
        upload.camera_make = metadata['camera_make']
        upload.camera_model = metadata['camera_model']
        upload.date_taken = metadata['date_taken']
        upload.location_latitude = metadata['latitude']
        upload.location_longitude = metadata['longitude']
        
        # Run weapon detection on the image
        if model:
            try:
                img = cv2.imread(image_path)
                if img is not None:
                    results = model(img)
                    
                    # Check if weapons detected
                    weapon_detected = False
                    max_confidence = 0
                    
                    for r in results:
                        if len(r.boxes) > 0:
                            for box in r.boxes:
                                conf = box.conf.item()
                                if conf > 0.5:  # Same threshold as live detection
                                    weapon_detected = True
                                    max_confidence = max(max_confidence, conf)
                    
                    upload.has_weapon = weapon_detected
                    upload.confidence = max_confidence
                    
                    # Save the annotated image if weapon detected
                    if weapon_detected:
                        # Create a detection record too
                        detection = Detection(
                            user=user,
                            image=upload.image.name,  # Use the same image file
                            confidence=max_confidence
                        )
                        detection.save()
                else:
                    print("Could not read image file")
            except Exception as e:
                print(f"Error during detection: {e}")
        
        upload.save()
        return upload
    except Exception as e:
        print(f"Error processing uploaded image: {e}")
        return None

@login_required
def image_upload(request):
    """Handle image upload form"""
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Process the uploaded image
            upload = process_uploaded_image(request.FILES['image'], request.user)
            if upload:
                # Redirect to the detail view
                return redirect('image_detail', pk=upload.id)
            else:
                messages.error(request, 'Error processing uploaded image.')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = ImageUploadForm()
    
    return render(request, 'image_upload.html', {'form': form})

@login_required
def image_detail(request, pk):
    """Show details for an uploaded image"""
    image = get_object_or_404(UploadedImage, pk=pk, user=request.user)
    return render(request, 'image_detail.html', {'image': image})

@login_required
def upload_history(request):
    """Show history of uploaded images"""
    uploads = UploadedImage.objects.filter(user=request.user).order_by('-uploaded_at')
    return render(request, 'upload_history.html', {'uploads': uploads})

# Cleanup function to be called when Django shuts down
def cleanup_all_cameras():
    """Clean up all active cameras"""
    global active_cameras, camera_lock
    
    with camera_lock:
        for user_id, handler in active_cameras.items():
            handler.cleanup()
        active_cameras.clear()
    print("All cameras cleaned up")