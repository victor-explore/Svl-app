from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import time
import random
import cv2
import atexit
import threading
import logging
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
@atexit.register
def cleanup():
    if camera_manager:
        camera_manager.shutdown()

# In-memory storage for cameras (replace with database in production)
cameras = []

def initialize_cameras():
    """Initialize all cameras in the enhanced camera manager"""
    for camera in cameras:
        # Add default fields for compatibility
        camera.setdefault('username', '')
        camera.setdefault('password', '')
        
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
    from database import db_manager
    
    # Get enriched detection records (50 latest) with camera info
    detections = db_manager.get_enriched_detection_history(limit=50)
    
    return render_template('sensor_analytics.html', detections=detections)

@app.route('/tracking')
def tracking():
    """Tracking page with timeline"""
    return render_template('tracking.html')


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
        
        # Create new camera
        new_camera = {
            'id': len(cameras) + 1,
            'name': data.get('name') or unique_id,  # Use unique_id as fallback display name
            'unique_id': unique_id,                 # Mandatory field
            'rtsp_url': data['rtsp_url'],
            'username': data.get('username', ''),
            'password': data.get('password', ''),
            'status': 'connecting' if data.get('auto_start', True) else 'offline',
            'auto_start': data.get('auto_start', True),
            'created_at': time.time()
        }
        
        cameras.append(new_camera)
        
        # Add camera to enhanced camera manager
        camera_manager.add_camera(new_camera)
        
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
    app.run(debug=True, host='0.0.0.0', port=5000)