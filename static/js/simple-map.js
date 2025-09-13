/**
 * Simple Offline Surveillance Map with Leaflet
 * Plots cameras using their lat/long coordinates on offline map tiles
 */

class SurveillanceMap {
    constructor(containerId, config) {
        this.containerId = containerId;
        this.config = config;
        this.map = null;
        this.markers = new Map();
        this.selectedCamera = null;
    }

    initialize() {
        try {
            this.initializeMap();
            this.loadCameras();
            console.log('Surveillance map initialized successfully');
        } catch (error) {
            console.error('Failed to initialize map:', error);
            this.showSetupModal();
        }
    }

    initializeMap() {
        // Initialize Leaflet map
        this.map = L.map(this.containerId, {
            center: this.config.center,
            zoom: this.config.zoom,
            minZoom: this.config.minZoom,
            maxZoom: this.config.maxZoom,
            zoomControl: true,
            attributionControl: false
        });

        // Add online tile layer
        const tileLayer = L.tileLayer(this.config.tileUrlTemplate, {
            attribution: '© OpenStreetMap contributors',
            maxZoom: 19
        });

        tileLayer.addTo(this.map);

        // Handle tile loading errors
        tileLayer.on('tileerror', (e) => {
            console.warn('Tile loading error:', e);
        });

        console.log('Map initialized with center:', this.config.center, 'zoom:', this.config.zoom);
    }


    async loadCameras() {
        try {
            const response = await fetch('/api/cameras');
            const data = await response.json();
            
            if (data.success) {
                this.updateCameraMarkers(data.cameras);
            } else {
                console.error('Failed to load cameras:', data.error);
            }
        } catch (error) {
            console.error('Error loading cameras:', error);
        }
    }

    updateCameraMarkers(cameras) {
        // Clear existing markers
        this.markers.forEach(marker => {
            this.map.removeLayer(marker);
        });
        this.markers.clear();

        // Add cameras with coordinates
        let camerasWithCoords = 0;
        cameras.forEach(camera => {
            if (camera.latitude != null && camera.longitude != null) {
                this.addCameraMarker(camera);
                camerasWithCoords++;
            }
        });

        console.log(`Added ${camerasWithCoords} cameras to map`);

        // Show warning if no cameras have coordinates
        if (camerasWithCoords === 0 && cameras.length > 0) {
            this.showNoCoordinatesWarning();
        }
    }

    addCameraMarker(camera) {
        const color = this.getCameraStatusColor(camera.status);
        
        // Create custom icon based on camera status
        const icon = L.divIcon({
            className: 'camera-marker',
            html: `<div style="
                background-color: ${color};
                width: 16px;
                height: 16px;
                border-radius: 50%;
                border: 2px solid white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.3);
                position: relative;
            ">
                <div style="
                    position: absolute;
                    top: -8px;
                    left: -8px;
                    width: 32px;
                    height: 32px;
                    border-radius: 50%;
                    background-color: ${color};
                    opacity: 0.3;
                    animation: pulse 2s infinite;
                "></div>
            </div>`,
            iconSize: [16, 16],
            iconAnchor: [8, 8]
        });

        // Create marker
        const marker = L.marker([camera.latitude, camera.longitude], {
            icon: icon,
            title: camera.name || camera.unique_id
        });

        // Add popup
        const popupContent = this.createMarkerPopup(camera);
        marker.bindPopup(popupContent, {
            maxWidth: 300,
            className: 'camera-popup'
        });

        // Handle marker click
        marker.on('click', () => {
            this.selectCamera(camera);
        });

        // Add to map
        marker.addTo(this.map);
        this.markers.set(camera.id, marker);
    }

    createMarkerPopup(camera) {
        const statusBadge = this.getStatusBadgeHtml(camera.status);
        return `
            <div class="p-2">
                <div class="font-semibold text-lg mb-2">${camera.name || camera.unique_id}</div>
                <div class="space-y-1 text-sm">
                    <div><strong>Status:</strong> ${statusBadge}</div>
                    <div><strong>ID:</strong> ${camera.unique_id}</div>
                    <div><strong>Coordinates:</strong> ${camera.latitude.toFixed(6)}, ${camera.longitude.toFixed(6)}</div>
                    ${camera.rtsp_url ? `<div><strong>Stream:</strong> <code class="text-xs">${camera.rtsp_url}</code></div>` : ''}
                </div>
            </div>
        `;
    }

    getCameraStatusColor(status) {
        return this.config.markerColors[status] || '#6b7280';
    }

    getStatusBadgeHtml(status) {
        const color = this.getCameraStatusColor(status);
        const statusText = status.charAt(0).toUpperCase() + status.slice(1);
        return `<span class="inline-flex items-center gap-1">
            <div style="width: 8px; height: 8px; background-color: ${color}; border-radius: 50%;"></div>
            ${statusText}
        </span>`;
    }

    selectCamera(camera) {
        this.selectedCamera = camera;
    }




    getSelectedCamera() {
        return this.selectedCamera;
    }

    showSetupModal() {
        const modal = document.getElementById('mapSetupModal');
        if (modal) {
            modal.showModal();
        }
    }

    showNoCoordinatesWarning() {
        const alertDiv = document.createElement('div');
        alertDiv.className = 'alert alert-warning fixed top-4 left-1/2 transform -translate-x-1/2 z-50 max-w-md shadow-lg';
        alertDiv.innerHTML = `
            <span>📍</span>
            <div>
                <p>No cameras have location coordinates set.</p>
                <p class="text-sm">Add latitude/longitude to cameras to see them on the map.</p>
            </div>
            <button type="button" class="btn btn-sm btn-circle btn-ghost" onclick="this.parentElement.remove()">✕</button>
        `;
        
        document.body.appendChild(alertDiv);
        
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 8000);
    }
}


// Add CSS for marker animations
const style = document.createElement('style');
style.textContent = `
    @keyframes pulse {
        0% { transform: scale(1); opacity: 0.3; }
        50% { transform: scale(1.2); opacity: 0.1; }
        100% { transform: scale(1); opacity: 0.3; }
    }
    
    .camera-popup .leaflet-popup-content {
        margin: 0;
    }
    
    .camera-marker {
        background: none !important;
        border: none !important;
    }
`;
document.head.appendChild(style);