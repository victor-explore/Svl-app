from flask import Flask, render_template, request, jsonify
import time
import random

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
        
        # Simulate connection test (replace with actual RTSP connection test)
        import time
        time.sleep(1)  # Simulate network delay
        
        # 70% success rate for demo
        success = random.random() > 0.3
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Connection successful! Camera stream is accessible.',
                'rtsp_url': rtsp_url
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Connection failed. Please check the RTSP URL and credentials.',
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)