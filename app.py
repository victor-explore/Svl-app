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
        'rtsp_url': 'rtsp://192.168.1.100:554/stream',
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

@app.route('/about')
def about():
    return render_template('about.html')

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
            cap = cv2.VideoCapture(rtsp_url)
            
            if not cap.isOpened():
                cap.release()
                return jsonify({
                    'success': False,
                    'error': 'Failed to open RTSP stream. Check URL and credentials.',
                    'rtsp_url': rtsp_url
                })
            
            # Try to read one frame to verify stream is working
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                return jsonify({
                    'success': True,
                    'message': 'Connection successful! Camera stream is accessible.',
                    'rtsp_url': rtsp_url
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'RTSP stream opened but no frames received. Check stream format.',
                    'rtsp_url': rtsp_url
                })
                
        except Exception as e:
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
    try:
        cap = cv2.VideoCapture(rtsp_url)
        
        if not cap.isOpened():
            cap.release()
            return None
        
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            # Convert BGR to RGB for web display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Encode as JPEG
            _, buffer = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return buffer.tobytes()
        
        return None
        
    except Exception as e:
        print(f"Error getting frame from {rtsp_url}: {e}")
        return None

def generate_video_stream(rtsp_url):
    """Generate video stream from RTSP camera"""
    cap = cv2.VideoCapture(rtsp_url)
    
    try:
        while True:
            if not cap.isOpened():
                break
                
            ret, frame = cap.read()
            
            if not ret or frame is None:
                break
            
            # Convert BGR to RGB for web display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 80])
            
            # Yield frame in multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                   
    except Exception as e:
        print(f"Error in video stream for {rtsp_url}: {e}")
    finally:
        if cap.isOpened():
            cap.release()

@app.route('/api/cameras/<int:camera_id>/stream')
def stream_camera(camera_id):
    """Stream video from RTSP camera"""
    try:
        # Find camera by ID
        camera = next((c for c in cameras if c['id'] == camera_id), None)
        if not camera:
            return jsonify({
                'success': False,
                'error': 'Camera not found'
            }), 404
        
        rtsp_url = camera['rtsp_url']
        
        return Response(
            generate_video_stream(rtsp_url),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)