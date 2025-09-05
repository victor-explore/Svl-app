from flask import Flask, render_template, request, jsonify, Response
import time
import random
import cv2
import atexit
import threading
from camera_manager import EnhancedCameraManager, CameraStatus
from config import *

app = Flask(__name__)

# Initialize enhanced camera manager
camera_manager = EnhancedCameraManager()

# Ensure cleanup on app shutdown
@atexit.register
def cleanup():
    if camera_manager:
        camera_manager.shutdown()

# In-memory storage for cameras (replace with database in production)
cameras = [
    {
        'id': 1,
        'name': 'Camera 1 - Front',
        'rtsp_url': 'rtsp://wowzaec2demo.streamlock.net/vod-multitrack/_definst_/mp4:ElephantsDream/ElephantsDream.mp4',
        'username': '',
        'password': '',
        'status': 'online',
        'auto_start': True,
        'created_at': time.time()
    },
    {
        'id': 2,
        'name': 'Camera 2 - Backyard',
        'rtsp_url': 'rtsp://192.168.1.101:554/stream',
        'username': '',
        'password': '',
        'status': 'online',
        'auto_start': True,
        'created_at': time.time()
    },
    {
        'id': 3,
        'name': 'Camera 3 - Garage',
        'rtsp_url': 'rtsp://192.168.1.102:554/stream',
        'username': '',
        'password': '',
        'status': 'offline',
        'auto_start': True,
        'created_at': time.time()
    },
    {
        'id': 4,
        'name': 'Camera 4 - Kitchen',
        'rtsp_url': 'rtsp://192.168.1.103:554/stream',
        'username': '',
        'password': '',
        'status': 'online',
        'auto_start': True,
        'created_at': time.time()
    },
    {
        'id': 5,
        'name': 'Camera 5 - Driveway',
        'rtsp_url': 'rtsp://192.168.1.104:554/stream',
        'username': '',
        'password': '',
        'status': 'connecting',
        'auto_start': True,
        'created_at': time.time()
    }
]

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
        if not data.get('name') or not data.get('rtsp_url'):
            return jsonify({
                'success': False,
                'error': 'Camera name and RTSP URL are required'
            }), 400
        
        # Create new camera
        new_camera = {
            'id': len(cameras) + 1,
            'name': data['name'],
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

@app.route('/api/cameras/<int:camera_id>', methods=['DELETE'])
def delete_camera(camera_id):
    """Delete a camera"""
    try:
        # Remove from camera manager first
        camera_manager.remove_camera(camera_id)
        
        # Remove from in-memory storage
        global cameras
        cameras = [c for c in cameras if c['id'] != camera_id]
        
        return jsonify({
            'success': True,
            'message': 'Camera deleted successfully'
        })
        
    except Exception as e:
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
    """Stream video from RTSP camera using enhanced camera manager"""
    print(f"[DEBUG] stream_camera endpoint called for camera_id: {camera_id}")
    try:
        # Check if camera exists
        camera = next((c for c in cameras if c['id'] == camera_id), None)
        if not camera:
            print(f"[ERROR] Camera with ID {camera_id} not found")
            return jsonify({
                'success': False,
                'error': 'Camera not found'
            }), 404
        
        print(f"[DEBUG] Found camera '{camera['name']}', starting enhanced stream...")
        
        # Use enhanced camera manager for streaming
        return Response(
            camera_manager.generate_video_stream(camera_id),
            mimetype='multipart/x-mixed-replace; boundary=frame'
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




@app.route('/api/cameras/<int:camera_id>/recordings', methods=['GET'])
def list_camera_recordings(camera_id):
    """List all recordings for a specific camera"""
    try:
        # Check if camera exists
        camera = next((c for c in cameras if c['id'] == camera_id), None)
        if not camera:
            return jsonify({
                'success': False,
                'error': 'Camera not found'
            }), 404
        
        recordings = camera_manager.list_recordings(camera_id)
        
        return jsonify({
            'success': True,
            'camera_id': camera_id,
            'camera_name': camera['name'],
            'recordings': recordings,
            'count': len(recordings)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/system/status', methods=['GET'])
def get_system_status():
    """Get overall system status"""
    try:
        all_statuses = camera_manager.get_all_camera_statuses()
        
        # Calculate system statistics
        total_cameras = len(cameras)
        online_cameras = sum(1 for status in all_statuses.values() if status and status['status'] == 'online')
        recording_cameras = sum(1 for status in all_statuses.values() if status and status['recording']['is_recording'])
        
        # Calculate total frames captured
        total_frames = sum(status['frames_captured'] for status in all_statuses.values() if status)
        
        return jsonify({
            'success': True,
            'system_status': {
                'total_cameras': total_cameras,
                'online_cameras': online_cameras,
                'offline_cameras': total_cameras - online_cameras,
                'recording_cameras': recording_cameras,
                'total_frames_captured': total_frames
            },
            'camera_statuses': all_statuses
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)