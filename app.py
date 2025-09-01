from flask import Flask, render_template, request, jsonify, Response
import time
import random
import cv2

app = Flask(__name__)

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
        'record_footage': False,
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
        'record_footage': False,
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
        'record_footage': False,
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
        'record_footage': False,
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
        'record_footage': False,
        'created_at': time.time()
    }
]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/feed')
def feed():
    # Count cameras by status
    stats = {
        'online': len([c for c in cameras if c['status'] == 'online']),
        'offline': len([c for c in cameras if c['status'] == 'offline']),
        'connecting': len([c for c in cameras if c['status'] == 'connecting'])
    }
    return render_template('feed.html', cameras=cameras, stats=stats)


@app.route('/api/cameras', methods=['GET'])
def get_cameras():
    """Get all cameras"""
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
            'record_footage': data.get('record_footage', False),
            'created_at': time.time()
        }
        
        cameras.append(new_camera)
        
        # Simulate status change after a delay (in real app, this would be handled by camera connection logic)
        if new_camera['auto_start']:
            # Simulate connection success/failure after 3-5 seconds
            import threading
            def update_status():
                time.sleep(random.uniform(3, 5))
                new_camera['status'] = 'online' if random.random() > 0.2 else 'offline'
            
            thread = threading.Thread(target=update_status)
            thread.daemon = True
            thread.start()
        
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
        
        # Real RTSP connection test using OpenCV
        try:
            print(f"[DEBUG] Testing connection to: {rtsp_url}")
            cap = cv2.VideoCapture(rtsp_url)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)  # 5 second timeout
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)   # 5 second read timeout
            
            if not cap.isOpened():
                print(f"[DEBUG] Connection test failed - could not open: {rtsp_url}")
                cap.release()
                return jsonify({
                    'success': False,
                    'error': 'Failed to open RTSP stream. Check URL and credentials.',
                    'rtsp_url': rtsp_url
                })
            
            print(f"[DEBUG] VideoCapture opened, trying to read test frame...")
            # Try to read one frame to verify stream is working
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                print(f"[DEBUG] Connection test SUCCESS - frame shape: {frame.shape}")
                return jsonify({
                    'success': True,
                    'message': 'Connection successful! Camera stream is accessible.',
                    'rtsp_url': rtsp_url
                })
            else:
                print(f"[DEBUG] Connection test FAILED - ret: {ret}, frame is None: {frame is None}")
                return jsonify({
                    'success': False,
                    'error': 'RTSP stream opened but no frames received. Check stream format.',
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

def get_camera_frame(rtsp_url):
    """Get a single frame from RTSP camera"""
    print(f"[DEBUG] get_camera_frame called with URL: {rtsp_url}")
    try:
        print(f"[DEBUG] Creating VideoCapture for {rtsp_url}")
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)  # 5 second timeout
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)   # 5 second read timeout
        
        if not cap.isOpened():
            print(f"[DEBUG] Failed to open VideoCapture for {rtsp_url}")
            cap.release()
            return None
        
        print(f"[DEBUG] VideoCapture opened successfully, reading frame...")
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            print(f"[DEBUG] Frame read successfully, shape: {frame.shape}")
            # Convert BGR to RGB for web display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Encode as JPEG
            _, buffer = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 80])
            print(f"[DEBUG] Frame encoded to JPEG, size: {len(buffer.tobytes())} bytes")
            return buffer.tobytes()
        
        print(f"[DEBUG] Failed to read frame: ret={ret}, frame is None: {frame is None}")
        return None
        
    except Exception as e:
        print(f"[ERROR] Exception in get_camera_frame from {rtsp_url}: {e}")
        return None

def generate_video_stream(rtsp_url):
    """Generate video stream from RTSP camera"""
    print(f"[DEBUG] generate_video_stream started for URL: {rtsp_url}")
    
    # Set timeout for OpenCV VideoCapture
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)  # 5 second timeout
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)   # 5 second read timeout
    
    try:
        frame_count = 0
        print(f"[DEBUG] Attempting to open VideoCapture with 5s timeout...")
        
        if not cap.isOpened():
            print(f"[ERROR] Failed to open VideoCapture in generate_video_stream for {rtsp_url}")
            return
        
        print(f"[DEBUG] VideoCapture opened successfully, starting streaming loop...")
        
        consecutive_failures = 0
        max_failures = 10
        
        while True:
            if not cap.isOpened():
                print(f"[DEBUG] VideoCapture closed unexpectedly, breaking loop")
                break
                
            ret, frame = cap.read()
            frame_count += 1
            
            if not ret or frame is None:
                consecutive_failures += 1
                print(f"[DEBUG] Failed to read frame {frame_count}: ret={ret}, frame is None: {frame is None}, consecutive failures: {consecutive_failures}")
                
                if consecutive_failures >= max_failures:
                    print(f"[ERROR] Too many consecutive failures ({consecutive_failures}), stopping stream")
                    break
                    
                time.sleep(0.1)  # Brief pause before retry
                continue
            
            consecutive_failures = 0  # Reset failure counter on success
            
            if frame_count % 30 == 1:  # Log every 30th frame to avoid spam
                print(f"[DEBUG] Successfully read frame {frame_count}, shape: {frame.shape}")
            
            # Convert BGR to RGB for web display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Encode frame as JPEG
            success, buffer = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 80])
            
            if not success:
                print(f"[ERROR] Failed to encode frame {frame_count} as JPEG")
                continue
            
            # Yield frame in multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                   
    except Exception as e:
        print(f"[ERROR] Exception in video stream for {rtsp_url}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"[DEBUG] Cleaning up VideoCapture for {rtsp_url}")
        if cap.isOpened():
            cap.release()

@app.route('/api/cameras/<int:camera_id>/stream')
def stream_camera(camera_id):
    """Stream video from RTSP camera"""
    print(f"[DEBUG] stream_camera endpoint called for camera_id: {camera_id}")
    try:
        # Find camera by ID
        camera = next((c for c in cameras if c['id'] == camera_id), None)
        if not camera:
            print(f"[ERROR] Camera with ID {camera_id} not found")
            return jsonify({
                'success': False,
                'error': 'Camera not found'
            }), 404
        
        rtsp_url = camera['rtsp_url']
        print(f"[DEBUG] Found camera '{camera['name']}' with URL: {rtsp_url}")
        
        print(f"[DEBUG] Starting Response with generate_video_stream...")
        return Response(
            generate_video_stream(rtsp_url),
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)