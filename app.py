from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import time
import random
import cv2
import atexit
import threading
import logging
import json
import os
from camera_manager import EnhancedCameraManager, CameraStatus
from config import *

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
SENIOR DEVELOPER CAMERA DELETION IMPLEMENTATION
================================================================
Problem: Camera deletion was taking 15+ seconds due to graceful thread/process cleanup
Solution: Optimistic UI + Asynchronous Background Cleanup

Key Features:
1. IMMEDIATE UI Response (<100ms) - User never waits
2. Progressive Timeout Strategy: [1s graceful] -> [2s terminate] -> [0s force_kill]
3. Background cleanup runs asynchronously in separate thread
4. Visual feedback shows "deleting" state for 300ms before removal
5. Error recovery - UI restored if backend deletion fails

Configuration (config.py):
- DELETE_STRATEGY = "optimistic" (vs "synchronous" for critical systems)  
- CLEANUP_TIMEOUT_PROGRESSIVE = [1, 2, 0] (graceful -> terminate -> force)
- SHOW_DELETION_FEEDBACK_MS = 300 (brief visual feedback duration)

Files Modified:
- app.py: Optimistic delete endpoint + progressive cleanup functions
- camera_manager.py: Three cleanup strategies (graceful/terminate/force)
- feed.html: Enhanced UI with deleting state and error recovery
- config.py: New configuration options
================================================================
"""

app = Flask(__name__)

# Initialize enhanced camera manager
camera_manager = EnhancedCameraManager()

# Ensure cleanup on app shutdown
# This should only be called when the entire application is terminating
@atexit.register
def cleanup():
    # Only shutdown if the camera manager exists and the app is truly exiting
    # This prevents premature shutdown during normal operations
    if camera_manager:
        import sys
        # Check if we're in a normal shutdown scenario
        # The atexit handler should only trigger during actual app termination
        logger.info("Application shutdown detected - cleaning up camera manager")
        camera_manager.shutdown()

# In-memory storage for cameras (replace with database in production)
cameras = []

def initialize_cameras():
    """Initialize all cameras in the enhanced camera manager"""
    global cameras
    
    # First, try to load cameras from database (for persistence across restarts)
    try:
        from database import db_manager, Camera
        db_manager.initialize()
        
        # Get all cameras from database
        session = db_manager.get_session()
        db_cameras = session.query(Camera).all()
        session.close()
        
        # If we have no in-memory cameras but have database cameras, restore them
        if len(cameras) == 0 and len(db_cameras) > 0:
            for db_camera in db_cameras:
                camera_dict = {
                    'id': db_camera.id,
                    'name': db_camera.name,
                    'unique_id': db_camera.unique_id,
                    'rtsp_url': db_camera.rtsp_url,
                    'latitude': db_camera.latitude,
                    'longitude': db_camera.longitude,
                    'username': '',  # Default values for manager compatibility
                    'password': '',
                    'status': 'offline',
                    'auto_start': True,
                    'created_at': db_camera.created_at.timestamp() if db_camera.created_at else time.time()
                }
                cameras.append(camera_dict)
                logger.info(f"Restored camera from database: {camera_dict['name']} (ID: {camera_dict['id']})")
                
    except Exception as e:
        logger.warning(f"Could not load cameras from database: {e}")
        # Continue with in-memory cameras only
    
    # Initialize all cameras in the camera manager
    for camera in cameras:
        # Add default fields for compatibility
        camera.setdefault('username', '')
        camera.setdefault('password', '')
        camera.setdefault('latitude', None)
        camera.setdefault('longitude', None)
        
        # Add to camera manager
        camera_manager.add_camera(camera)
        
        # Update status based on camera manager
        camera_status = camera_manager.get_camera_status(camera['id'])
        if camera_status:
            camera['status'] = camera_status['status']

# Initialize cameras on startup
initialize_cameras()

@app.route('/')
def home():
    # Update camera statuses from camera manager
    for camera in cameras:
        camera_status = camera_manager.get_camera_status(camera['id'])
        if camera_status:
            camera['status'] = camera_status['status']
    
    # Count cameras by status
    stats = {
        'online': len([c for c in cameras if c['status'] == 'online']),
        'offline': len([c for c in cameras if c['status'] == 'offline']),
        'connecting': len([c for c in cameras if c['status'] == 'connecting'])
    }
    return render_template('feed.html', cameras=cameras, stats=stats)

@app.route('/feed')
def feed():
    # Update camera statuses from camera manager
    for camera in cameras:
        camera_status = camera_manager.get_camera_status(camera['id'])
        if camera_status:
            camera['status'] = camera_status['status']
    
    # Count cameras by status
    stats = {
        'online': len([c for c in cameras if c['status'] == 'online']),
        'offline': len([c for c in cameras if c['status'] == 'offline']),
        'connecting': len([c for c in cameras if c['status'] == 'connecting'])
    }
    return render_template('feed.html', cameras=cameras, stats=stats)

@app.route('/sensor-analytics')
def sensor_analytics():
    """Sensor Analytics page to display detection database entries"""
    import math
    from datetime import datetime
    from database import db_manager
    
    # Get pagination parameters from URL
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    
    # Get date filter parameters from URL
    start_date_str = request.args.get('start_date')
    end_date_str = request.args.get('end_date')
    
    # Parse and validate date parameters
    start_date = None
    end_date = None
    
    if start_date_str:
        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        except ValueError:
            start_date = None
    
    if end_date_str:
        try:
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
            # Set end time to end of day (23:59:59)
            end_date = end_date.replace(hour=23, minute=59, second=59)
        except ValueError:
            end_date = None
    
    # Ensure valid values
    page = max(1, page)
    per_page = max(1, min(per_page, 100))  # Limit max to 100 for performance
    
    # Calculate offset
    offset = (page - 1) * per_page
    
    # Get paginated detection records with camera info and date filters
    detections = db_manager.get_enriched_detection_history(
        limit=per_page, 
        offset=offset,
        start_date=start_date,
        end_date=end_date
    )
    
    # Get total count for pagination with date filters
    total_records = db_manager.get_total_detection_count(
        start_date=start_date,
        end_date=end_date
    )
    
    # Calculate pagination info
    total_pages = math.ceil(total_records / per_page) if total_records > 0 else 1
    
    pagination = {
        'page': page,
        'per_page': per_page,
        'total': total_records,
        'pages': total_pages,
        'has_prev': page > 1,
        'has_next': page < total_pages,
        'prev_page': page - 1 if page > 1 else None,
        'next_page': page + 1 if page < total_pages else None
    }
    
    # Prepare date filter info for template
    date_filter = {
        'start_date': start_date_str or '',
        'end_date': end_date_str or ''
    }
    
    return render_template('sensor_analytics.html', 
                         detections=detections, 
                         pagination=pagination,
                         date_filter=date_filter)

@app.route('/tracking')
def tracking():
    """Tracking page with timeline"""
    return render_template('tracking.html')

@app.route('/settings')
def settings():
    """Settings page for runtime configuration"""
    return render_template('settings.html')

@app.route('/map')
def map_page():
    """Map page showing camera locations"""
    config_data = {
        'MAP_DEFAULT_CENTER_LAT': MAP_DEFAULT_CENTER_LAT,
        'MAP_DEFAULT_CENTER_LNG': MAP_DEFAULT_CENTER_LNG,
        'MAP_DEFAULT_ZOOM': MAP_DEFAULT_ZOOM,
        'MAP_MIN_ZOOM': MAP_MIN_ZOOM,
        'MAP_MAX_ZOOM': MAP_MAX_ZOOM,
        'MAP_TILE_URL_TEMPLATE': MAP_TILE_URL_TEMPLATE,
        'CAMERA_MARKER_COLORS': CAMERA_MARKER_COLORS
    }
    return render_template('map.html', config=config_data)

# Settings management utilities
SETTINGS_FILE = 'user_settings.json'

def load_user_settings():
    """Load user settings from file, return empty dict if not found"""
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.error(f"Error loading user settings: {e}")
        return {}

def save_user_settings(settings):
    """Save user settings to file"""
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving user settings: {e}")
        return False

def get_runtime_settings():
    """Get current runtime settings (merged defaults + user overrides)"""
    # Define runtime-changeable settings with their current values
    user_settings = load_user_settings()
    
    runtime_settings = {
        # RTSP Connection Settings
        'rtsp_timeout_ms': user_settings.get('rtsp_timeout_ms', RTSP_TIMEOUT_MS),
        'rtsp_read_timeout_ms': user_settings.get('rtsp_read_timeout_ms', RTSP_READ_TIMEOUT_MS),
        'rtsp_reconnect_delay': user_settings.get('rtsp_reconnect_delay', RTSP_RECONNECT_DELAY),
        'rtsp_max_reconnect_attempts': user_settings.get('rtsp_max_reconnect_attempts', RTSP_MAX_RECONNECT_ATTEMPTS),
        'rtsp_reconnect_delay_max': user_settings.get('rtsp_reconnect_delay_max', RTSP_RECONNECT_DELAY_MAX),
        
        # Frame Processing Settings  
        'frame_queue_size': user_settings.get('frame_queue_size', FRAME_QUEUE_SIZE),
        'processing_fps': user_settings.get('processing_fps', PROCESSING_FPS),
        'jpeg_quality': user_settings.get('jpeg_quality', JPEG_QUALITY),
        
        # Performance Settings
        'max_cameras': user_settings.get('max_cameras', MAX_CAMERAS),
        'thread_cleanup_timeout': user_settings.get('thread_cleanup_timeout', THREAD_CLEANUP_TIMEOUT),
        'status_update_interval': user_settings.get('status_update_interval', STATUS_UPDATE_INTERVAL),
        
        # Camera Management Settings
        'delete_strategy': user_settings.get('delete_strategy', DELETE_STRATEGY),
        'show_deletion_feedback_ms': user_settings.get('show_deletion_feedback_ms', SHOW_DELETION_FEEDBACK_MS),
        
        # Detection Settings
        'person_detection_enabled': user_settings.get('person_detection_enabled', PERSON_DETECTION_ENABLED),
        'person_detection_confidence': user_settings.get('person_detection_confidence', PERSON_DETECTION_CONFIDENCE),
        'person_detection_interval': user_settings.get('person_detection_interval', PERSON_DETECTION_INTERVAL),
        'person_detection_draw_boxes': user_settings.get('person_detection_draw_boxes', PERSON_DETECTION_DRAW_BOXES),
        'person_detection_resize_enabled': user_settings.get('person_detection_resize_enabled', PERSON_DETECTION_RESIZE_ENABLED),
        'person_detection_resize_width': user_settings.get('person_detection_resize_width', PERSON_DETECTION_RESIZE_WIDTH),
        'person_detection_resize_height': user_settings.get('person_detection_resize_height', PERSON_DETECTION_RESIZE_HEIGHT),
        
        # Database & Storage Settings
        'database_enabled': user_settings.get('database_enabled', DATABASE_ENABLED),
        'detection_image_storage_enabled': user_settings.get('detection_image_storage_enabled', DETECTION_IMAGE_STORAGE_ENABLED),
        'detection_image_quality': user_settings.get('detection_image_quality', DETECTION_IMAGE_QUALITY),
        'database_cleanup_enabled': user_settings.get('database_cleanup_enabled', DATABASE_CLEANUP_ENABLED),
        'database_cleanup_days': user_settings.get('database_cleanup_days', DATABASE_CLEANUP_DAYS),
        'detection_storage_interval_seconds': user_settings.get('detection_storage_interval_seconds', DETECTION_STORAGE_INTERVAL_SECONDS),
    }
    
    return runtime_settings

def apply_runtime_settings(settings):
    """Apply settings that can be changed at runtime"""
    # Update global variables that affect new operations
    global RTSP_TIMEOUT_MS, RTSP_READ_TIMEOUT_MS, RTSP_RECONNECT_DELAY
    global RTSP_MAX_RECONNECT_ATTEMPTS, RTSP_RECONNECT_DELAY_MAX
    global FRAME_QUEUE_SIZE, PROCESSING_FPS, JPEG_QUALITY
    global MAX_CAMERAS, THREAD_CLEANUP_TIMEOUT, STATUS_UPDATE_INTERVAL
    global DELETE_STRATEGY, SHOW_DELETION_FEEDBACK_MS
    global PERSON_DETECTION_ENABLED, PERSON_DETECTION_CONFIDENCE, PERSON_DETECTION_INTERVAL
    global PERSON_DETECTION_DRAW_BOXES, PERSON_DETECTION_RESIZE_ENABLED
    global PERSON_DETECTION_RESIZE_WIDTH, PERSON_DETECTION_RESIZE_HEIGHT
    global DATABASE_ENABLED, DETECTION_IMAGE_STORAGE_ENABLED, DETECTION_IMAGE_QUALITY
    global DATABASE_CLEANUP_ENABLED, DATABASE_CLEANUP_DAYS, DETECTION_STORAGE_INTERVAL_SECONDS
    
    # RTSP Settings (affect new connections)
    RTSP_TIMEOUT_MS = settings['rtsp_timeout_ms']
    RTSP_READ_TIMEOUT_MS = settings['rtsp_read_timeout_ms'] 
    RTSP_RECONNECT_DELAY = settings['rtsp_reconnect_delay']
    RTSP_MAX_RECONNECT_ATTEMPTS = settings['rtsp_max_reconnect_attempts']
    RTSP_RECONNECT_DELAY_MAX = settings['rtsp_reconnect_delay_max']
    
    # Frame Processing (can affect existing cameras through camera manager)
    FRAME_QUEUE_SIZE = settings['frame_queue_size']
    PROCESSING_FPS = settings['processing_fps']
    JPEG_QUALITY = settings['jpeg_quality']
    
    # Performance Settings
    MAX_CAMERAS = settings['max_cameras']
    THREAD_CLEANUP_TIMEOUT = settings['thread_cleanup_timeout']
    STATUS_UPDATE_INTERVAL = settings['status_update_interval']
    
    # Camera Management
    DELETE_STRATEGY = settings['delete_strategy']
    SHOW_DELETION_FEEDBACK_MS = settings['show_deletion_feedback_ms']
    
    # Person Detection (can be updated in detection system)
    PERSON_DETECTION_ENABLED = settings['person_detection_enabled']
    PERSON_DETECTION_CONFIDENCE = settings['person_detection_confidence']
    PERSON_DETECTION_INTERVAL = settings['person_detection_interval']
    PERSON_DETECTION_DRAW_BOXES = settings['person_detection_draw_boxes']
    PERSON_DETECTION_RESIZE_ENABLED = settings['person_detection_resize_enabled']
    PERSON_DETECTION_RESIZE_WIDTH = settings['person_detection_resize_width']
    PERSON_DETECTION_RESIZE_HEIGHT = settings['person_detection_resize_height']
    
    # Database & Storage
    DATABASE_ENABLED = settings['database_enabled']
    DETECTION_IMAGE_STORAGE_ENABLED = settings['detection_image_storage_enabled']
    DETECTION_IMAGE_QUALITY = settings['detection_image_quality']
    DATABASE_CLEANUP_ENABLED = settings['database_cleanup_enabled']
    DATABASE_CLEANUP_DAYS = settings['database_cleanup_days']
    DETECTION_STORAGE_INTERVAL_SECONDS = settings['detection_storage_interval_seconds']

@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Get current runtime settings"""
    try:
        settings = get_runtime_settings()
        return jsonify({
            'success': True,
            'settings': settings
        })
    except Exception as e:
        logger.error(f"Error getting settings: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/settings', methods=['PUT'])
def update_settings():
    """Update runtime settings"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No settings data provided'
            }), 400
        
        # Validate settings
        current_settings = get_runtime_settings()
        
        # Only allow updating known runtime settings
        valid_keys = set(current_settings.keys())
        invalid_keys = set(data.keys()) - valid_keys
        
        if invalid_keys:
            return jsonify({
                'success': False,
                'error': f'Invalid setting keys: {", ".join(invalid_keys)}'
            }), 400
        
        # Basic type and range validation
        validation_errors = []
        
        # Numeric validations
        numeric_ranges = {
            'rtsp_timeout_ms': (1000, 30000),
            'rtsp_read_timeout_ms': (1000, 15000),
            'rtsp_reconnect_delay': (1, 10),
            'rtsp_max_reconnect_attempts': (1, 50),
            'rtsp_reconnect_delay_max': (5, 300),
            'frame_queue_size': (1, 50),
            'processing_fps': (1, 60),
            'jpeg_quality': (10, 100),
            'max_cameras': (1, 100),
            'thread_cleanup_timeout': (1, 30),
            'status_update_interval': (1, 10),
            'show_deletion_feedback_ms': (100, 5000),
            'person_detection_confidence': (0.1, 1.0),
            'person_detection_interval': (1, 120),
            'person_detection_resize_width': (320, 1920),
            'person_detection_resize_height': (240, 1080),
            'detection_image_quality': (50, 100),
            'database_cleanup_days': (1, 365),
            'detection_storage_interval_seconds': (1, 300)
        }
        
        for key, value in data.items():
            if key in numeric_ranges:
                try:
                    num_val = float(value)
                    min_val, max_val = numeric_ranges[key]
                    if not (min_val <= num_val <= max_val):
                        validation_errors.append(f'{key} must be between {min_val} and {max_val}')
                except (ValueError, TypeError):
                    validation_errors.append(f'{key} must be a number')
            
            elif key in ['person_detection_enabled', 'person_detection_draw_boxes', 
                        'person_detection_resize_enabled', 'database_enabled', 
                        'detection_image_storage_enabled', 'database_cleanup_enabled']:
                if not isinstance(value, bool):
                    validation_errors.append(f'{key} must be a boolean')
            
            elif key == 'delete_strategy':
                if value not in ['optimistic', 'synchronous']:
                    validation_errors.append('delete_strategy must be either "optimistic" or "synchronous"')
        
        if validation_errors:
            return jsonify({
                'success': False,
                'error': 'Validation errors: ' + '; '.join(validation_errors)
            }), 400
        
        # Load current user settings
        user_settings = load_user_settings()
        
        # Update with new values
        user_settings.update(data)
        
        # Save to file
        if not save_user_settings(user_settings):
            return jsonify({
                'success': False,
                'error': 'Failed to save settings'
            }), 500
        
        # Apply runtime changes
        updated_settings = get_runtime_settings()
        apply_runtime_settings(updated_settings)
        
        return jsonify({
            'success': True,
            'message': 'Settings updated successfully',
            'settings': updated_settings
        })
        
    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/settings/reset', methods=['POST'])
def reset_settings():
    """Reset all settings to defaults"""
    try:
        # Remove user settings file
        if os.path.exists(SETTINGS_FILE):
            os.remove(SETTINGS_FILE)
        
        # Get default settings
        default_settings = get_runtime_settings()
        
        # Apply defaults
        apply_runtime_settings(default_settings)
        
        return jsonify({
            'success': True,
            'message': 'Settings reset to defaults successfully',
            'settings': default_settings
        })
        
    except Exception as e:
        logger.error(f"Error resetting settings: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/cameras', methods=['GET'])
def get_cameras():
    """Get all cameras with real-time status"""
    # Update camera statuses from camera manager
    for camera in cameras:
        camera_status = camera_manager.get_camera_status(camera['id'])
        if camera_status:
            camera['status'] = camera_status['status']
            camera['fps'] = camera_status.get('fps', 0)
            camera['frames_captured'] = camera_status.get('frames_captured', 0)
    
    return jsonify({
        'success': True,
        'cameras': cameras,
        'count': len(cameras)
    })

@app.route('/api/cameras', methods=['POST'])
def add_camera():
    """Add a new camera"""
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data.get('unique_id') or not data.get('rtsp_url'):
            return jsonify({
                'success': False,
                'error': 'Unique ID and RTSP URL are required'
            }), 400

        # Validate unique_id format
        import re
        unique_id = data.get('unique_id', '').strip()
        if not re.match(r'^[a-zA-Z0-9_-]+$', unique_id):
            return jsonify({
                'success': False,
                'error': 'Unique ID can only contain letters, numbers, underscores, and hyphens'
            }), 400

        # Check for duplicate RTSP URL
        existing_camera = next((c for c in cameras if c['rtsp_url'] == data['rtsp_url']), None)
        if existing_camera:
            return jsonify({
                'success': False,
                'error': f'RTSP URL already exists for camera "{existing_camera["name"]}"'
            }), 400
        
        # Validate latitude and longitude if provided
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        
        if latitude is not None:
            try:
                latitude = float(latitude)
                if not -90 <= latitude <= 90:
                    return jsonify({
                        'success': False,
                        'error': 'Latitude must be between -90 and 90 degrees'
                    }), 400
            except (ValueError, TypeError):
                return jsonify({
                    'success': False,
                    'error': 'Invalid latitude value'
                }), 400
                
        if longitude is not None:
            try:
                longitude = float(longitude)
                if not -180 <= longitude <= 180:
                    return jsonify({
                        'success': False,
                        'error': 'Longitude must be between -180 and 180 degrees'
                    }), 400
            except (ValueError, TypeError):
                return jsonify({
                    'success': False,
                    'error': 'Invalid longitude value'
                }), 400
        
        # Create new camera
        new_camera = {
            'id': len(cameras) + 1,
            'name': data.get('name') or unique_id,  # Use unique_id as fallback display name
            'unique_id': unique_id,                 # Mandatory field
            'rtsp_url': data['rtsp_url'],
            'username': data.get('username', ''),
            'password': data.get('password', ''),
            'latitude': latitude,
            'longitude': longitude,
            'status': 'connecting' if data.get('auto_start', True) else 'offline',
            'auto_start': data.get('auto_start', True),
            'created_at': time.time()
        }
        
        cameras.append(new_camera)
        
        # Add camera to enhanced camera manager
        camera_manager.add_camera(new_camera)
        
        # Also persist camera to database for map coordinates and detection records
        try:
            from database import db_manager
            db_manager.create_or_get_camera(
                camera_id=new_camera['id'],
                camera_name=new_camera['name'],
                camera_unique_id=new_camera['unique_id'],
                rtsp_url=new_camera['rtsp_url'],
                latitude=new_camera.get('latitude'),
                longitude=new_camera.get('longitude')
            )
        except Exception as db_error:
            logger.warning(f"Failed to persist camera to database: {db_error}")
            # Continue without failing the API call
        
        return jsonify({
            'success': True,
            'camera': new_camera,
            'message': 'Camera added successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/cameras/test-connection', methods=['POST'])
def test_camera_connection():
    """Test RTSP camera connection"""
    try:
        data = request.get_json()
        rtsp_url = data.get('rtsp_url')
        username = data.get('username', '')
        password = data.get('password', '')
        
        if not rtsp_url:
            return jsonify({
                'success': False,
                'error': 'RTSP URL is required'
            }), 400
        
        # Enhanced RTSP connection test using camera worker
        try:
            print(f"[DEBUG] Testing connection to: {rtsp_url}")
            
            # Create a temporary camera worker for testing
            from camera_manager import CameraWorker
            test_worker = CameraWorker(
                camera_id=999,  # Temporary ID
                name="Test Camera",
                rtsp_url=rtsp_url,
                username=username,
                password=password
            )
            
            test_worker.start()
            
            # Wait a few seconds for connection attempt
            connection_timeout = 10
            start_time = time.time()
            
            while time.time() - start_time < connection_timeout:
                status = test_worker.status
                if status == CameraStatus.ONLINE:
                    # Try to get a frame
                    frame, timestamp = test_worker.get_latest_frame()
                    test_worker.stop()
                    
                    if frame is not None:
                        print(f"[DEBUG] Connection test SUCCESS - frame shape: {frame.shape}")
                        return jsonify({
                            'success': True,
                            'message': 'Connection successful! Camera stream is accessible.',
                            'rtsp_url': rtsp_url,
                            'frame_shape': frame.shape
                        })
                    else:
                        return jsonify({
                            'success': False,
                            'error': 'Connected but no frames received yet. Stream may need more time.',
                            'rtsp_url': rtsp_url
                        })
                        
                elif status == CameraStatus.ERROR:
                    test_worker.stop()
                    stats = test_worker.get_stats()
                    return jsonify({
                        'success': False,
                        'error': f'Connection failed: {stats.get("last_error", "Unknown error")}',
                        'rtsp_url': rtsp_url
                    })
                
                time.sleep(0.5)  # Check every 500ms
            
            # Timeout reached
            test_worker.stop()
            return jsonify({
                'success': False,
                'error': 'Connection test timeout. Camera may be unreachable.',
                'rtsp_url': rtsp_url
            })
                
        except Exception as e:
            print(f"[ERROR] Connection test exception for {rtsp_url}: {e}")
            return jsonify({
                'success': False,
                'error': f'Connection error: {str(e)}',
                'rtsp_url': rtsp_url
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def cleanup_camera_resources(camera_id):
    """Background cleanup with progressive timeout strategy - Senior Developer Approach"""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Starting background cleanup for camera {camera_id}")
        
        # Progressive timeout approach: graceful -> terminate -> force_kill
        cleanup_strategies = list(zip(CLEANUP_TIMEOUT_PROGRESSIVE, ["graceful", "terminate", "force_kill"]))
        
        for timeout, method in cleanup_strategies:
            logger.info(f"Attempting {method} cleanup for camera {camera_id} (timeout: {timeout}s)")
            
            if _attempt_cleanup(camera_id, timeout, method):
                logger.info(f"Camera {camera_id} successfully cleaned up using {method} method")
                return
        
        # If all strategies failed
        logger.error(f"Failed to cleanup camera {camera_id} after all attempts - resources may be leaked")
        
    except Exception as e:
        logger.error(f"Background cleanup failed for camera {camera_id}: {e}")

def _attempt_cleanup(camera_id, timeout, method):
    """Attempt camera resource cleanup with specific strategy"""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        if method == "graceful":
            # Standard graceful cleanup with short timeout
            return camera_manager.remove_camera_graceful(camera_id, timeout)
        
        elif method == "terminate":
            # More aggressive cleanup with process termination
            return camera_manager.remove_camera_terminate(camera_id, timeout)
        
        elif method == "force_kill":
            # Force kill all processes immediately
            return camera_manager.remove_camera_force(camera_id)
        
    except Exception as e:
        logger.error(f"Cleanup attempt {method} failed for camera {camera_id}: {e}")
        return False
    
    return False

@app.route('/api/cameras/<int:camera_id>', methods=['DELETE'])
def delete_camera(camera_id):
    """Delete a camera - Optimistic UI approach for immediate user feedback"""
    global cameras
    
    logger.info(f"DELETE request received for camera_id: {camera_id}")
    logger.info(f"Current cameras before deletion: {[c['id'] for c in cameras]}")
    
    try:
        # 1. Validate camera exists
        camera = next((c for c in cameras if c['id'] == camera_id), None)
        if not camera:
            logger.warning(f"Camera {camera_id} not found for deletion")
            return jsonify({
                'success': False,
                'error': 'Camera not found'
            }), 404
        
        camera_name = camera['name']
        logger.info(f"Deleting camera: {camera_name} (ID: {camera_id})")
        
        # 2. Remove from UI list immediately (optimistic approach)
        cameras_before_count = len(cameras)
        cameras = [c for c in cameras if c['id'] != camera_id]
        cameras_after_count = len(cameras)
        
        logger.info(f"Camera count: {cameras_before_count} -> {cameras_after_count}")
        logger.info(f"Remaining cameras after deletion: {[c['id'] for c in cameras]}")
        
        # 3. Schedule background cleanup (non-blocking)
        if DELETE_STRATEGY == "optimistic":
            cleanup_thread = threading.Thread(
                target=cleanup_camera_resources, 
                args=(camera_id,), 
                daemon=True,
                name=f"cleanup_camera_{camera_id}"
            )
            cleanup_thread.start()
            logger.info(f"Scheduled background cleanup for camera {camera_id} ({camera_name})")
        else:
            # Fallback to synchronous cleanup for critical systems
            camera_manager.remove_camera(camera_id)
        
        # 4. Return immediate success to user
        return jsonify({
            'success': True,
            'message': f'Camera "{camera_name}" deleted successfully'
        })
        
    except Exception as e:
        logger.error(f"Error in delete_camera endpoint: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/cameras/<int:camera_id>/status', methods=['PUT'])
def update_camera_status(camera_id):
    """Update camera status"""
    try:
        data = request.get_json()
        new_status = data.get('status')
        
        if new_status not in ['online', 'offline', 'connecting']:
            return jsonify({
                'success': False,
                'error': 'Invalid status. Must be one of: online, offline, connecting'
            }), 400
        
        # Find and update camera
        camera = next((c for c in cameras if c['id'] == camera_id), None)
        if not camera:
            return jsonify({
                'success': False,
                'error': 'Camera not found'
            }), 404
        
        camera['status'] = new_status
        
        return jsonify({
            'success': True,
            'camera': camera,
            'message': f'Camera status updated to {new_status}'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def get_camera_frame(camera_id):
    """Get a single frame from camera using enhanced camera manager"""
    print(f"[DEBUG] get_camera_frame called for camera_id: {camera_id}")
    try:
        frame_bytes = camera_manager.get_camera_frame(camera_id)
        if frame_bytes:
            print(f"[DEBUG] Frame retrieved successfully, size: {len(frame_bytes)} bytes")
            return frame_bytes
        else:
            print(f"[DEBUG] No frame available for camera {camera_id}")
            return None
    except Exception as e:
        print(f"[ERROR] Exception in get_camera_frame for camera {camera_id}: {e}")
        return None

@app.route('/api/cameras/<int:camera_id>/stream')
def stream_camera(camera_id):
    """Stream video from RTSP camera using enhanced camera manager with person detection"""
    print(f"[DEBUG] stream_camera endpoint called for camera_id: {camera_id}")
    try:
        # Check if camera exists
        camera = next((c for c in cameras if c['id'] == camera_id), None)
        if not camera:
            # Log at debug level to reduce noise
            logger.debug(f"Stream request for non-existent camera {camera_id}")
            return jsonify({
                'success': False,
                'error': 'Camera not found'
            }), 404
        
        print(f"[DEBUG] Found camera '{camera['name']}', starting enhanced stream...")
        
        # Check if detection visualization is requested
        draw_detections = request.args.get('detections', 'true').lower() == 'true'
        
        # Use enhanced camera manager for streaming with detection support
        return Response(
            camera_manager.generate_video_stream(camera_id, draw_detections),
            mimetype='multipart/x-mixed-replace; boundary=frame',
            headers={
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0'
            }
        )
        
    except Exception as e:
        print(f"[ERROR] Exception in stream_camera for camera_id {camera_id}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# New Enhanced API Endpoints

@app.route('/api/status/stream')
def status_stream():
    """Server-Sent Events for real-time camera status updates"""
    import json
    
    def event_stream():
        logger.info("[SSE] Client connected to status stream")
        try:
            while True:
                # Get current status of all cameras
                cameras_status = []
                for camera in cameras:
                    camera_status = camera_manager.get_camera_status(camera['id'])
                    if camera_status:
                        cameras_status.append({
                            'id': camera['id'],
                            'name': camera['name'],
                            'status': camera_status['status'],
                            'frames_captured': camera_status.get('frames_captured', 0),
                            'fps': camera_status.get('fps', 0)
                        })
                    else:
                        # Camera manager doesn't have this camera, mark as offline
                        cameras_status.append({
                            'id': camera['id'],
                            'name': camera['name'],
                            'status': 'offline',
                            'frames_captured': 0,
                            'fps': 0
                        })
                
                # Send status update
                data = json.dumps({
                    'type': 'camera_status',
                    'timestamp': time.time(),
                    'cameras': cameras_status
                })
                yield f"data: {data}\n\n"
                
                # Update every 1 second for real-time responsiveness
                time.sleep(1)
                
        except GeneratorExit:
            logger.info("[SSE] Client disconnected from status stream")
        except Exception as e:
            logger.error(f"[SSE] Error in status stream: {e}")
    
    return Response(stream_with_context(event_stream()),
                   mimetype='text/event-stream',
                   headers={
                       'Cache-Control': 'no-cache',
                       'Connection': 'keep-alive',
                       'X-Accel-Buffering': 'no'  # Disable nginx buffering
                   })

@app.route('/api/system/status', methods=['GET'])
def get_system_status():
    """Get overall system status"""
    try:
        all_statuses = camera_manager.get_all_camera_statuses()
        
        # Calculate system statistics
        total_cameras = len(cameras)
        online_cameras = sum(1 for status in all_statuses.values() if status and status['status'] == 'online')
        
        # Calculate total frames captured
        total_frames = sum(status['frames_captured'] for status in all_statuses.values() if status)
        
        return jsonify({
            'success': True,
            'system_status': {
                'total_cameras': total_cameras,
                'online_cameras': online_cameras,
                'offline_cameras': total_cameras - online_cameras,
                'total_frames_captured': total_frames
            },
            'camera_statuses': all_statuses
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Person Detection API Endpoints

@app.route('/api/cameras/<int:camera_id>/detections', methods=['GET'])
def get_camera_detections(camera_id):
    """Get current person detection data for a camera"""
    try:
        # Import detection manager here to avoid circular imports
        from person_detector import detection_manager
        
        # Check if camera exists
        camera = next((c for c in cameras if c['id'] == camera_id), None)
        if not camera:
            # Log at debug level to reduce noise from frontend polling deleted cameras
            logger.debug(f"Detection request for non-existent camera {camera_id}")
            return jsonify({
                'success': False,
                'error': 'Camera not found'
            }), 404
        
        # Get detection statistics
        detection_stats = detection_manager.get_detection_stats(camera_id)
        if not detection_stats:
            return jsonify({
                'success': False,
                'error': 'Detection data not available for this camera'
            }), 404
        
        # Get recent detections
        limit = int(request.args.get('limit', 10))
        recent_detections = detection_manager.get_recent_detections(camera_id, limit)
        
        # Convert detections to dictionary format
        detection_data = [detection.to_dict() for detection in recent_detections]
        
        return jsonify({
            'success': True,
            'camera_id': camera_id,
            'detection_stats': detection_stats,
            'recent_detections': detection_data,
            'detection_count': len(detection_data)
        })
        
    except Exception as e:
        logger.error(f"Error getting detections for camera {camera_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/cameras/<int:camera_id>/detection-settings', methods=['GET', 'PUT'])
def camera_detection_settings(camera_id):
    """Get or update detection settings for a camera"""
    try:
        from person_detector import detection_manager
        
        # Check if camera exists
        camera = next((c for c in cameras if c['id'] == camera_id), None)
        if not camera:
            return jsonify({
                'success': False,
                'error': 'Camera not found'
            }), 404
        
        if request.method == 'GET':
            # Get current detection settings
            detection_stats = detection_manager.get_detection_stats(camera_id)
            if not detection_stats:
                return jsonify({
                    'success': False,
                    'error': 'Detection settings not available for this camera'
                }), 404
            
            return jsonify({
                'success': True,
                'camera_id': camera_id,
                'settings': {
                    'detection_enabled': detection_stats['detection_enabled'],
                    'confidence_threshold': detection_stats['confidence_threshold'],
                    'model_path': detection_stats['model_path']
                }
            })
        
        elif request.method == 'PUT':
            # Update detection settings
            data = request.get_json()
            if not data:
                return jsonify({
                    'success': False,
                    'error': 'No data provided'
                }), 400
            
            # Update detection enabled/disabled status
            if 'detection_enabled' in data:
                enabled = bool(data['detection_enabled'])
                detection_manager.enable_detection(camera_id, enabled)
            
            # Update confidence threshold
            if 'confidence_threshold' in data:
                try:
                    threshold = float(data['confidence_threshold'])
                    if 0.0 <= threshold <= 1.0:
                        detector = detection_manager.get_detector(camera_id)
                        detector.set_confidence_threshold(threshold)
                    else:
                        return jsonify({
                            'success': False,
                            'error': 'Confidence threshold must be between 0.0 and 1.0'
                        }), 400
                except (ValueError, TypeError):
                    return jsonify({
                        'success': False,
                        'error': 'Invalid confidence threshold value'
                    }), 400
            
            return jsonify({
                'success': True,
                'message': 'Detection settings updated successfully'
            })
    
    except Exception as e:
        logger.error(f"Error handling detection settings for camera {camera_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/detections/summary', methods=['GET'])
def get_detection_summary():
    """Get detection summary for all cameras"""
    try:
        from person_detector import detection_manager
        
        summary = {
            'total_cameras': len(cameras),
            'detection_enabled_cameras': 0,
            'total_persons_detected': 0,
            'cameras_with_recent_activity': 0,
            'camera_summaries': {}
        }
        
        for camera in cameras:
            camera_id = camera['id']
            detection_stats = detection_manager.get_detection_stats(camera_id)
            
            if detection_stats:
                is_enabled = detection_stats['detection_enabled']
                total_detections = detection_stats.get('total_detections', 0)
                recent_detections = len(detection_manager.get_recent_detections(camera_id, 1))
                
                if is_enabled:
                    summary['detection_enabled_cameras'] += 1
                
                summary['total_persons_detected'] += total_detections
                
                if recent_detections > 0:
                    summary['cameras_with_recent_activity'] += 1
                
                summary['camera_summaries'][camera_id] = {
                    'camera_name': camera['name'],
                    'detection_enabled': is_enabled,
                    'total_detections': total_detections,
                    'recent_activity': recent_detections > 0,
                    'last_detection_time': detection_stats.get('last_detection_time')
                }
        
        return jsonify({
            'success': True,
            'summary': summary
        })
        
    except Exception as e:
        logger.error(f"Error getting detection summary: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/detection/status', methods=['GET'])
def get_detection_status():
    """Get global detection system status and model initialization info"""
    try:
        from person_detector import detection_manager
        
        # Import config values
        from config import (PERSON_DETECTION_ENABLED, PERSON_DETECTION_MODEL, 
                          PERSON_DETECTION_CONFIDENCE, PERSON_DETECTION_INTERVAL)
        
        status = {
            'detection_enabled_globally': PERSON_DETECTION_ENABLED,
            'detection_model': PERSON_DETECTION_MODEL,
            'detection_confidence': PERSON_DETECTION_CONFIDENCE,
            'detection_interval': PERSON_DETECTION_INTERVAL,
            'cameras_with_detection': {},
            'total_cameras': len(cameras)
        }
        
        # Get status for each camera
        for camera in cameras:
            camera_id = camera['id']
            camera_status = {
                'camera_name': camera['name'],
                'has_detector': camera_id in detection_manager.detectors,
                'detection_enabled': detection_manager.is_detection_enabled(camera_id),
                'model_initialized': False,
                'model_status': 'Not created'
            }
            
            if camera_id in detection_manager.detectors:
                detector = detection_manager.detectors[camera_id]
                camera_status['model_initialized'] = detector.is_initialized
                camera_status['model_status'] = 'Initialized' if detector.is_initialized else 'Not initialized'
                camera_status['model_path'] = detector.model_path
                camera_status['confidence_threshold'] = detector.confidence_threshold
                camera_status['detector_stats'] = detector.get_stats()
                
            status['cameras_with_detection'][camera_id] = camera_status
        
        return jsonify({
            'success': True,
            'detection_system_status': status
        })
        
    except Exception as e:
        logger.error(f"Error getting detection status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Database Storage API Endpoints

@app.route('/api/detections/history', methods=['GET'])
def get_detection_history():
    """Get detection history with optional filters"""
    try:
        # Import database manager only when needed
        from config import DATABASE_ENABLED
        if not DATABASE_ENABLED:
            return jsonify({
                'success': False,
                'error': 'Database storage is not enabled'
            }), 400
        
        from database import db_manager
        
        # Get query parameters
        camera_id = request.args.get('camera_id', type=int)
        limit = request.args.get('limit', 50, type=int)
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # Parse date parameters if provided
        start_datetime = None
        end_datetime = None
        if start_date:
            try:
                start_datetime = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            except ValueError:
                return jsonify({
                    'success': False,
                    'error': 'Invalid start_date format. Use ISO format (YYYY-MM-DDTHH:MM:SS)'
                }), 400
        
        if end_date:
            try:
                end_datetime = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            except ValueError:
                return jsonify({
                    'success': False,
                    'error': 'Invalid end_date format. Use ISO format (YYYY-MM-DDTHH:MM:SS)'
                }), 400
        
        # Get detection history
        detections = db_manager.get_detection_history(
            camera_id=camera_id,
            limit=limit,
            start_date=start_datetime,
            end_date=end_datetime
        )
        
        # Convert to dictionary format
        detection_data = [detection.to_dict() for detection in detections]
        
        return jsonify({
            'success': True,
            'detections': detection_data,
            'total_returned': len(detection_data),
            'filters': {
                'camera_id': camera_id,
                'limit': limit,
                'start_date': start_date,
                'end_date': end_date
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting detection history: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/detections/images/<path:image_path>')
def serve_detection_image(image_path):
    """Serve detection images from storage"""
    try:
        from config import DETECTION_IMAGE_STORAGE_ENABLED
        if not DETECTION_IMAGE_STORAGE_ENABLED:
            return jsonify({
                'success': False,
                'error': 'Image storage is not enabled'
            }), 400
        
        from detection_storage import image_storage
        from flask import send_file
        
        # Get full path to image
        full_path = image_storage.get_image_path(image_path)
        
        # Check if file exists
        if not image_storage.image_exists(image_path):
            return jsonify({
                'success': False,
                'error': 'Image not found'
            }), 404
        
        # Serve the image file
        return send_file(full_path, mimetype='image/jpeg')
        
    except Exception as e:
        logger.error(f"Error serving detection image {image_path}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/detections/stats', methods=['GET'])
def get_detection_statistics():
    """Get detection statistics from database"""
    try:
        from config import DATABASE_ENABLED
        if not DATABASE_ENABLED:
            return jsonify({
                'success': False,
                'error': 'Database storage is not enabled'
            }), 400
        
        from database import db_manager
        
        camera_id = request.args.get('camera_id', type=int)
        
        # Get detection statistics
        stats = db_manager.get_detection_stats(camera_id)
        
        # Get additional system statistics
        system_stats = {}
        if not camera_id:  # Get system-wide stats only if no specific camera requested
            try:
                # Get stats for all cameras
                all_stats = []
                for camera in cameras:
                    cam_stats = db_manager.get_detection_stats(camera['id'])
                    cam_stats['camera_name'] = camera['name']
                    all_stats.append(cam_stats)
                
                system_stats = {
                    'total_cameras': len(cameras),
                    'cameras_with_detections': len([s for s in all_stats if s['total_detections'] > 0]),
                    'system_total_detections': sum(s['total_detections'] for s in all_stats),
                    'camera_stats': all_stats
                }
            except Exception as e:
                logger.warning(f"Error getting system stats: {e}")
        
        return jsonify({
            'success': True,
            'detection_stats': stats,
            'system_stats': system_stats
        })
        
    except Exception as e:
        logger.error(f"Error getting detection statistics: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/analytics/hourly-detections', methods=['GET'])
def get_hourly_detections():
    """Get hourly detection statistics for analytics chart"""
    try:
        from config import DATABASE_ENABLED
        if not DATABASE_ENABLED:
            return jsonify({
                'success': False,
                'error': 'Database functionality is not enabled'
            }), 400

        from database import db_manager
        from datetime import datetime
        
        # Get date range parameters
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')
        hours_back = request.args.get('hours_back', default=24, type=int)
        
        # Parse date parameters if provided
        start_date = None
        end_date = None
        
        if start_date_str and end_date_str:
            try:
                start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
                end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
                # Set end time to end of day (23:59:59)
                end_date = end_date.replace(hour=23, minute=59, second=59)
                
                # Basic validation
                if start_date > end_date:
                    return jsonify({
                        'success': False,
                        'error': 'Start date cannot be after end date'
                    }), 400
                    
            except ValueError:
                return jsonify({
                    'success': False,
                    'error': 'Invalid date format. Use YYYY-MM-DD'
                }), 400
        else:
            # Limit hours_back to reasonable range when using default behavior
            hours_back = max(1, min(hours_back, 168))  # 1 hour to 1 week
        
        # Get statistics using appropriate method
        if start_date and end_date:
            stats = db_manager.get_hourly_detection_stats(start_date=start_date, end_date=end_date)
        else:
            stats = db_manager.get_hourly_detection_stats(hours_back=hours_back)
        
        return jsonify({
            'success': True,
            'data': stats,
            'range_type': 'date_range' if start_date and end_date else 'hours_back',
            'parameters': {
                'start_date': start_date_str,
                'end_date': end_date_str,
                'hours_back': hours_back if not (start_date and end_date) else None
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting hourly detection statistics: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/detections/export', methods=['GET'])
def export_detections():
    """Export detection data in CSV or JSON format"""
    try:
        from config import DATABASE_ENABLED
        if not DATABASE_ENABLED:
            return jsonify({
                'success': False,
                'error': 'Database storage is not enabled'
            }), 400
        
        from database import db_manager
        import csv
        import io
        
        # Get query parameters
        format_type = request.args.get('format', 'json').lower()
        camera_id = request.args.get('camera_id', type=int)
        limit = request.args.get('limit', 1000, type=int)
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        if format_type not in ['json', 'csv']:
            return jsonify({
                'success': False,
                'error': 'Format must be either "json" or "csv"'
            }), 400
        
        # Parse date parameters if provided
        start_datetime = None
        end_datetime = None
        if start_date:
            try:
                start_datetime = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            except ValueError:
                return jsonify({
                    'success': False,
                    'error': 'Invalid start_date format'
                }), 400
        
        if end_date:
            try:
                end_datetime = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            except ValueError:
                return jsonify({
                    'success': False,
                    'error': 'Invalid end_date format'
                }), 400
        
        # Get detection data
        detections = db_manager.get_detection_history(
            camera_id=camera_id,
            limit=limit,
            start_date=start_datetime,
            end_date=end_datetime
        )
        
        if format_type == 'json':
            # Export as JSON
            detection_data = [detection.to_dict() for detection in detections]
            return jsonify({
                'success': True,
                'detections': detection_data,
                'total_exported': len(detection_data)
            })
        
        else:  # CSV export
            # Create CSV content
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow([
                'Detection ID', 'Person ID', 'Camera ID', 'Camera Name',
                'Confidence', 'BBox X1', 'BBox Y1', 'BBox X2', 'BBox Y2',
                'Detection Time', 'Image Path'
            ])
            
            # Write detection data
            for detection in detections:
                writer.writerow([
                    detection.id,
                    detection.person_id,
                    detection.camera_id,
                    detection.camera.name if detection.camera else 'Unknown',
                    detection.confidence,
                    detection.bbox_x1,
                    detection.bbox_y1,
                    detection.bbox_x2,
                    detection.bbox_y2,
                    detection.created_at.isoformat(),
                    detection.image_path or ''
                ])
            
            # Return CSV file
            csv_content = output.getvalue()
            output.close()
            
            return Response(
                csv_content,
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment; filename=detections_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'}
            )
        
    except Exception as e:
        logger.error(f"Error exporting detections: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/detections/storage-stats', methods=['GET'])
def get_storage_statistics():
    """Get image storage statistics"""
    try:
        from config import DETECTION_IMAGE_STORAGE_ENABLED, DATABASE_ENABLED
        
        stats = {
            'database_enabled': DATABASE_ENABLED,
            'image_storage_enabled': DETECTION_IMAGE_STORAGE_ENABLED
        }
        
        # Get database stats
        if DATABASE_ENABLED:
            try:
                from database import db_manager
                db_stats = db_manager.get_detection_stats()
                stats['database_stats'] = db_stats
            except Exception as e:
                logger.warning(f"Error getting database stats: {e}")
                stats['database_stats'] = {'error': str(e)}
        
        # Get image storage stats
        if DETECTION_IMAGE_STORAGE_ENABLED:
            try:
                from detection_storage import image_storage
                storage_stats = image_storage.get_storage_stats()
                stats['image_storage_stats'] = storage_stats
            except Exception as e:
                logger.warning(f"Error getting image storage stats: {e}")
                stats['image_storage_stats'] = {'error': str(e)}
        
        return jsonify({
            'success': True,
            'storage_statistics': stats
        })
        
    except Exception as e:
        logger.error(f"Error getting storage statistics: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    # IMPORTANT: Debug mode controlled by config to prevent app shutdown when all cameras are removed
    # The auto-reloader in debug mode can cause unexpected shutdowns
    # Use debug=False for production-like stability
    from config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG
    app.run(debug=FLASK_DEBUG, host=FLASK_HOST, port=FLASK_PORT, threaded=True)