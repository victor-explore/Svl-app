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

        // Handle marker click - show modal instead of popup
        marker.on('click', () => {
            this.selectCamera(camera);
            this.showCameraModal(camera);
        });

        // Add to map
        marker.addTo(this.map);
        this.markers.set(camera.id, marker);
    }

    showCameraModal(camera) {
        const modal = document.getElementById('cameraModal');
        const content = document.getElementById('cameraModalContent');

        const statusBadge = this.getStatusBadgeHtml(camera.status);

        content.innerHTML = `
            <h3 class="text-2xl font-bold mb-4">${camera.name || camera.unique_id}</h3>

            <!-- Video Feed -->
            <div class="aspect-video bg-base-300 rounded-lg overflow-hidden mb-4">
                <img src="/api/cameras/${camera.id}/stream"
                     alt="Camera ${camera.unique_id} Stream"
                     class="w-full h-full object-contain"
                     onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';" />
                <div class="w-full h-full flex items-center justify-center text-base-content/60" style="display: none;">
                    <div class="text-center">
                        <div class="text-4xl mb-2">📷</div>
                        <div>Stream unavailable</div>
                    </div>
                </div>
            </div>

            <!-- Detections Chart -->
            <div class="mt-4">
                <div class="bg-base-200 rounded-lg p-4">
                    <div style="position: relative; height: 350px; width: 100%;">
                        <canvas id="cameraDetectionsChart_${camera.id}"></canvas>
                    </div>
                </div>
            </div>

            <!-- Camera Information -->
            <div class="space-y-2 text-sm">
                <div class="flex justify-between">
                    <span class="font-medium">Status:</span>
                    <span>${statusBadge}</span>
                </div>
                <div class="flex justify-between">
                    <span class="font-medium">ID:</span>
                    <span>${camera.unique_id}</span>
                </div>
                <div class="flex justify-between">
                    <span class="font-medium">Coordinates:</span>
                    <span>${camera.latitude.toFixed(6)}, ${camera.longitude.toFixed(6)}</span>
                </div>
            </div>
        `;

        modal.showModal();

        // Initialize chart after modal is shown
        setTimeout(() => {
            this.initializeCameraChart(camera);
        }, 100);
    }

    initializeCameraChart(camera) {
        const canvasId = `cameraDetectionsChart_${camera.id}`;
        const canvas = document.getElementById(canvasId);

        if (!canvas) {
            console.error('Chart canvas not found:', canvasId);
            return;
        }

        const ctx = canvas.getContext('2d');

        // Create chart
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Detections',
                    data: [],
                    borderColor: '#3b82f6',
                    backgroundColor: '#3b82f620',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Hour'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Detections'
                        },
                        beginAtZero: true,
                        ticks: {
                            precision: 0
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });

        // Fetch and load data
        this.loadCameraChartData(camera.id, chart);
    }

    loadCameraChartData(cameraId, chart) {
        fetch(`/api/analytics/camera-${cameraId}/hourly-detections?hours_back=24`)
            .then(response => response.json())
            .then(result => {
                if (result.success && result.data) {
                    chart.data.labels = result.data.hours;
                    chart.data.datasets[0].data = result.data.counts;
                    chart.update();
                } else {
                    console.error('Error loading camera chart data:', result.error);
                }
            })
            .catch(error => {
                console.error('Error loading camera chart data:', error);
            });
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

    .camera-marker {
        background: none !important;
        border: none !important;
    }
`;
document.head.appendChild(style);