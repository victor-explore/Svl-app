#!/usr/bin/env python3
"""
Simple Map Tile Downloader for Offline Surveillance Maps

This script downloads OpenStreetMap tiles for a specific area to enable
offline map functionality with accurate lat/long coordinate plotting.
"""

import os
import requests
import math
import sys
from pathlib import Path

class MapTileDownloader:
    def __init__(self, base_dir="static/map_tiles"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.tile_server = "https://tile.openstreetmap.org"
        
    def deg2num(self, lat_deg, lon_deg, zoom):
        """Convert lat/lon to tile numbers"""
        lat_rad = math.radians(lat_deg)
        n = 2.0 ** zoom
        x = int((lon_deg + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return (x, y)
    
    def download_tile(self, x, y, zoom):
        """Download a single map tile"""
        url = f"{self.tile_server}/{zoom}/{x}/{y}.png"
        tile_dir = self.base_dir / str(zoom) / str(x)
        tile_dir.mkdir(parents=True, exist_ok=True)
        tile_path = tile_dir / f"{y}.png"
        
        if tile_path.exists():
            return True  # Already downloaded
        
        try:
            headers = {
                'User-Agent': 'SVL Surveillance System/1.0'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            with open(tile_path, 'wb') as f:
                f.write(response.content)
            return True
        except Exception as e:
            print(f"Failed to download tile {zoom}/{x}/{y}: {e}")
            return False
    
    def download_area(self, center_lat, center_lon, radius_km=2, zoom_levels=None):
        """Download tiles for a circular area around center point"""
        if zoom_levels is None:
            zoom_levels = [12, 13, 14, 15]  # Good balance of detail vs storage
        
        print(f"Downloading map tiles for area:")
        print(f"  Center: {center_lat:.6f}, {center_lon:.6f}")
        print(f"  Radius: {radius_km} km")
        print(f"  Zoom levels: {zoom_levels}")
        
        total_tiles = 0
        downloaded = 0
        
        for zoom in zoom_levels:
            # Calculate approximate tile bounds for the area
            # This is a rough approximation - tiles at edges might be outside radius
            lat_offset = radius_km / 111.0  # Rough km to degrees conversion
            lon_offset = radius_km / (111.0 * math.cos(math.radians(center_lat)))
            
            # Get tile bounds
            min_x, max_y = self.deg2num(center_lat - lat_offset, center_lon - lon_offset, zoom)
            max_x, min_y = self.deg2num(center_lat + lat_offset, center_lon + lon_offset, zoom)
            
            print(f"\nZoom level {zoom}:")
            zoom_tiles = (max_x - min_x + 1) * (max_y - min_y + 1)
            total_tiles += zoom_tiles
            print(f"  Downloading {zoom_tiles} tiles...")
            
            for x in range(min_x, max_x + 1):
                for y in range(min_y, max_y + 1):
                    if self.download_tile(x, y, zoom):
                        downloaded += 1
                    
                    if downloaded % 50 == 0:
                        print(f"  Progress: {downloaded}/{total_tiles} tiles")
        
        print(f"\nDownload complete!")
        print(f"Downloaded {downloaded} tiles")
        print(f"Storage location: {self.base_dir.absolute()}")
        
        # Calculate approximate storage size
        try:
            storage_mb = sum(f.stat().st_size for f in self.base_dir.rglob('*.png')) / (1024 * 1024)
            print(f"Storage used: {storage_mb:.1f} MB")
        except:
            print("Storage size calculation failed")

def main():
    print("=== SVL Surveillance Map Setup ===\n")
    
    # Get user input
    try:
        print("Enter the center coordinates of your surveillance area:")
        center_lat = float(input("Latitude (e.g., 40.7589): "))
        center_lon = float(input("Longitude (e.g., -73.9851): "))
        
        print("\nEnter surveillance area radius:")
        radius_km = float(input("Radius in kilometers (default 2): ") or "2")
        
        print(f"\nThis will download map tiles for:")
        print(f"  Center: {center_lat:.6f}, {center_lon:.6f}")
        print(f"  Radius: {radius_km} km")
        print(f"  Estimated storage: {radius_km * radius_km * 10:.0f} MB")
        
        confirm = input("\nProceed with download? (y/N): ").lower()
        if confirm != 'y':
            print("Setup cancelled.")
            return
        
    except (ValueError, KeyboardInterrupt):
        print("\nSetup cancelled.")
        return
    
    # Download tiles
    downloader = MapTileDownloader()
    downloader.download_area(center_lat, center_lon, radius_km)
    
    print(f"\n=== Next Steps ===")
    print(f"1. Update config.py with these coordinates:")
    print(f"   MAP_CENTER_LAT = {center_lat}")
    print(f"   MAP_CENTER_LNG = {center_lon}")
    print(f"2. Add lat/lng coordinates to your cameras")
    print(f"3. Access the map at http://localhost:5000/map")

if __name__ == "__main__":
    main()