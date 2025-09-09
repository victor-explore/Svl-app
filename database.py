"""
Database models and operations for CCTV Person Detection Storage
Handles persistent storage of detection records and camera metadata
"""

import os
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
import logging

logger = logging.getLogger(__name__)

def get_ist_now():
    """Get current datetime using local system time (IST)"""
    return datetime.now()

Base = declarative_base()

class Camera(Base):
    """Camera metadata table"""
    __tablename__ = 'cameras'
    
    id = Column(Integer, primary_key=True, index=True)
    unique_id = Column(String(50), index=True)
    name = Column(String(200), nullable=False)
    rtsp_url = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: get_ist_now().replace(tzinfo=None))
    
    # Relationship to detections
    detections = relationship("Detection", back_populates="camera")

class Detection(Base):
    """Person detection records table"""
    __tablename__ = 'detections'
    
    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(String(50), index=True)  # UUID for unique person identification
    camera_id = Column(Integer, ForeignKey('cameras.id'), nullable=False)
    
    # Detection details
    confidence = Column(Float, nullable=False)
    bbox_x1 = Column(Float, nullable=False)
    bbox_y1 = Column(Float, nullable=False)
    bbox_x2 = Column(Float, nullable=False)
    bbox_y2 = Column(Float, nullable=False)
    
    # File paths
    image_path = Column(Text, nullable=False)  # Path to detection image
    
    # Timestamps
    created_at = Column(DateTime, default=lambda: get_ist_now().replace(tzinfo=None))
    
    # Additional metadata
    frame_width = Column(Integer)
    frame_height = Column(Integer)
    
    # Relationship
    camera = relationship("Camera", back_populates="detections")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert detection to dictionary for API responses"""
        return {
            'id': self.id,
            'person_id': self.person_id,
            'camera_id': self.camera_id,
            'camera_unique_id': self.camera.unique_id,
            'camera_name': self.camera.name,
            'confidence': round(self.confidence, 3),
            'bbox': [self.bbox_x1, self.bbox_y1, self.bbox_x2, self.bbox_y2],
            'image_path': self.image_path,
            'detected_at': self.created_at.isoformat(),
            'created_at': self.created_at.isoformat(),
            'frame_dimensions': [self.frame_width, self.frame_height] if self.frame_width else None
        }

class DatabaseManager:
    """Manages database operations for person detection storage"""
    
    def __init__(self, database_url: str = "sqlite:///detection_records.db"):
        """Initialize database manager with SQLite database"""
        self.database_url = database_url
        self.engine = None
        self.SessionLocal = None
        self._initialized = False
        
    def initialize(self):
        """Initialize database connection and create tables"""
        try:
            # Create database directory if it doesn't exist
            if self.database_url.startswith("sqlite:///"):
                db_path = self.database_url.replace("sqlite:///", "")
                db_dir = os.path.dirname(db_path)
                if db_dir and not os.path.exists(db_dir):
                    os.makedirs(db_dir)
            
            # Create engine and session factory
            self.engine = create_engine(
                self.database_url,
                connect_args={"check_same_thread": False} if "sqlite" in self.database_url else {}
            )
            
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            
            # Create all tables
            Base.metadata.create_all(bind=self.engine)
            
            self._initialized = True
            logger.info(f"Database initialized successfully: {self.database_url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def get_session(self) -> Session:
        """Get database session"""
        if not self._initialized:
            self.initialize()
        return self.SessionLocal()
    
    def create_or_get_camera(self, camera_id: int, camera_name: str, 
                           camera_unique_id: str = None, rtsp_url: str = "") -> Camera:
        """Create or get camera record"""
        session = self.get_session()
        try:
            # Try to find existing camera
            camera = session.query(Camera).filter(Camera.id == camera_id).first()
            
            if not camera:
                # Create new camera record
                if not camera_unique_id:
                    camera_unique_id = f"camera_{camera_id}_{int(get_ist_now().timestamp())}"
                
                camera = Camera(
                    id=camera_id,
                    unique_id=camera_unique_id,
                    name=camera_name,
                    rtsp_url=rtsp_url,
                    created_at=get_ist_now().replace(tzinfo=None)
                )
                session.add(camera)
                session.commit()
                logger.info(f"Created new camera record: {camera_name} (ID: {camera_id})")
            else:
                # Update camera info if needed
                camera.name = camera_name
                session.commit()
                logger.debug(f"Updated camera record: {camera_name} (ID: {camera_id})")
            
            # Access all attributes while session is active to prevent lazy loading errors
            camera_id_val = camera.id
            camera_name_val = camera.name
            camera_unique_id_val = camera.unique_id
            
            # Expunge object from session so it can be used after session closes
            session.expunge(camera)
            return camera
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating/getting camera: {e}")
            raise
        finally:
            session.close()
    
    def save_detection(self, camera_id: int, 
                      detection_data: Dict[str, Any]) -> Detection:
        """Save person detection to database"""
        session = self.get_session()
        try:
            # Handle camera creation/retrieval within THIS session to avoid detached instances
            camera = session.query(Camera).filter(Camera.id == camera_id).first()
            
            if not camera:
                # Create new camera record in the same session
                camera_unique_id = f"camera_{camera_id}_{int(get_ist_now().timestamp())}"
                
                camera = Camera(
                    id=camera_id,
                    unique_id=camera_unique_id,
                    name=detection_data.get('camera_name', f'Camera {camera_id}'),
                    rtsp_url=detection_data.get('rtsp_url', ''),
                    created_at=get_ist_now().replace(tzinfo=None)
                )
                session.add(camera)
                session.flush()  # Ensure camera gets ID without committing yet
                logger.info(f"Created new camera record: {camera.name} (ID: {camera_id})")
            else:
                # Update existing camera info in the same session
                if 'camera_name' in detection_data:
                    camera.name = detection_data['camera_name']
                logger.debug(f"Updated camera record: {camera.name} (ID: {camera_id})")
            
            # Generate unique person ID
            person_id = str(uuid.uuid4())
            
            # Create detection record using the attached camera object
            detection = Detection(
                person_id=person_id,
                camera_id=camera_id,
                confidence=detection_data['confidence'],
                bbox_x1=detection_data['bbox'][0],
                bbox_y1=detection_data['bbox'][1],
                bbox_x2=detection_data['bbox'][2],
                bbox_y2=detection_data['bbox'][3],
                image_path=detection_data['image_path'],
                frame_width=detection_data.get('frame_width'),
                frame_height=detection_data.get('frame_height')
            )
            
            session.add(detection)
            session.commit()  # Commit both camera and detection in single transaction
            
            # Access attributes while session is active to prevent lazy loading errors
            detection_id = detection.id
            detection_person_id = detection.person_id
            
            # Expunge object from session so it can be used after session closes
            session.expunge(detection)
            
            logger.info(f"Saved detection: Person {person_id} in camera {camera.name}")
            return detection
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving detection: {e}")
            raise
        finally:
            session.close()
    
    def get_detection_history(self, camera_id: int = None, limit: int = 50, 
                            start_date: datetime = None, end_date: datetime = None) -> List[Detection]:
        """Get detection history with optional filters"""
        session = self.get_session()
        try:
            query = session.query(Detection)
            
            # Apply filters
            if camera_id:
                query = query.filter(Detection.camera_id == camera_id)
            
            if start_date:
                query = query.filter(Detection.created_at >= start_date)
            
            if end_date:
                query = query.filter(Detection.created_at <= end_date)
            
            # Order by most recent first and limit results
            detections = query.order_by(Detection.created_at.desc()).limit(limit).all()
            
            return detections
            
        except Exception as e:
            logger.error(f"Error getting detection history: {e}")
            return []
        finally:
            session.close()
    
    def get_enriched_detection_history(self, limit: int = 50, offset: int = 0,
                                     start_date: datetime = None, end_date: datetime = None) -> List[Dict[str, Any]]:
        """Get detection history with camera information properly joined as dictionaries"""
        session = self.get_session()
        try:
            # Step 1: Get all cameras with their basic info
            cameras = session.query(Camera).all()
            camera_lookup = {cam.id: {'name': cam.name, 'unique_id': cam.unique_id} for cam in cameras}
            
            # Step 2: Build detection query with filters
            query = session.query(Detection)
            
            if start_date:
                query = query.filter(Detection.created_at >= start_date)
            
            if end_date:
                query = query.filter(Detection.created_at <= end_date)
            
            # Get detections ordered by most recent first with offset and limit
            detections = query.order_by(Detection.created_at.desc()).offset(offset).limit(limit).all()
            
            # Step 3: Combine data manually into dictionaries
            enriched_detections = []
            for detection in detections:
                camera_info = camera_lookup.get(
                    detection.camera_id, 
                    {'name': f'Camera {detection.camera_id}', 'unique_id': f'camera_{detection.camera_id}'}
                )
                
                enriched_detections.append({
                    'id': detection.id,
                    'person_id': detection.person_id,
                    'camera_id': detection.camera_id,
                    'camera_name': camera_info['name'],
                    'camera_unique_id': camera_info['unique_id'],
                    'confidence': detection.confidence,
                    'bbox': [detection.bbox_x1, detection.bbox_y1, detection.bbox_x2, detection.bbox_y2],
                    'image_path': detection.image_path,
                    'created_at': detection.created_at,
                    'frame_width': detection.frame_width,
                    'frame_height': detection.frame_height
                })
            
            return enriched_detections
            
        except Exception as e:
            logger.error(f"Error getting enriched detection history: {e}")
            return []
        finally:
            session.close()
    
    def get_total_detection_count(self, start_date: datetime = None, end_date: datetime = None) -> int:
        """Get total count of detection records for pagination"""
        session = self.get_session()
        try:
            query = session.query(Detection)
            
            if start_date:
                query = query.filter(Detection.created_at >= start_date)
            
            if end_date:
                query = query.filter(Detection.created_at <= end_date)
            
            count = query.count()
            return count
            
        except Exception as e:
            logger.error(f"Error getting total detection count: {e}")
            return 0
        finally:
            session.close()

    def get_detections_by_unique_id(self, camera_unique_id: str, limit: int = 50, 
                                  start_date: datetime = None, end_date: datetime = None) -> List[Detection]:
        """Get all detections for cameras with the specified unique_id (aggregated across all camera records)"""
        session = self.get_session()
        try:
            # Get all camera IDs with this unique_id
            camera_ids_query = session.query(Camera.id).filter(Camera.unique_id == camera_unique_id)
            camera_id_list = [cam_id[0] for cam_id in camera_ids_query.all()]
            
            if not camera_id_list:
                return []
            
            # Get detections for all these cameras
            query = session.query(Detection).filter(Detection.camera_id.in_(camera_id_list))
            
            # Apply date filters
            if start_date:
                query = query.filter(Detection.created_at >= start_date)
            
            if end_date:
                query = query.filter(Detection.created_at <= end_date)
            
            # Order by most recent first and limit results
            detections = query.order_by(Detection.created_at.desc()).limit(limit).all()
            
            return detections
            
        except Exception as e:
            logger.error(f"Error getting detections by unique_id {camera_unique_id}: {e}")
            return []
        finally:
            session.close()
    
    def get_detection_stats(self, camera_id: int = None) -> Dict[str, Any]:
        """Get detection statistics"""
        session = self.get_session()
        try:
            query = session.query(Detection)
            
            if camera_id:
                query = query.filter(Detection.camera_id == camera_id)
            
            total_detections = query.count()
            
            # Get recent detections (last 24 hours)
            recent_time = get_ist_now().replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
            recent_detections = query.filter(Detection.created_at >= recent_time).count()
            
            # Get latest detection
            latest_detection = query.order_by(Detection.created_at.desc()).first()
            
            stats = {
                'total_detections': total_detections,
                'recent_detections_today': recent_detections,
                'latest_detection_time': latest_detection.created_at.isoformat() if latest_detection else None,
                'camera_id': camera_id
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting detection stats: {e}")
            return {'total_detections': 0, 'recent_detections_today': 0}
        finally:
            session.close()
    
    def get_detection_stats_by_unique_id(self, camera_unique_id: str) -> Dict[str, Any]:
        """Get detection statistics aggregated for all cameras with the specified unique_id"""
        session = self.get_session()
        try:
            # Get all camera IDs with this unique_id
            camera_ids_query = session.query(Camera.id).filter(Camera.unique_id == camera_unique_id)
            camera_id_list = [cam_id[0] for cam_id in camera_ids_query.all()]
            
            if not camera_id_list:
                return {
                    'total_detections': 0,
                    'recent_detections_today': 0,
                    'latest_detection_time': None,
                    'camera_unique_id': camera_unique_id,
                    'camera_records_count': 0
                }
            
            # Get all detections for cameras with this unique_id
            query = session.query(Detection).filter(Detection.camera_id.in_(camera_id_list))
            
            total_detections = query.count()
            
            # Get recent detections (last 24 hours)
            recent_time = get_ist_now().replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
            recent_detections = query.filter(Detection.created_at >= recent_time).count()
            
            # Get latest detection
            latest_detection = query.order_by(Detection.created_at.desc()).first()
            
            stats = {
                'total_detections': total_detections,
                'recent_detections_today': recent_detections,
                'latest_detection_time': latest_detection.created_at.isoformat() if latest_detection else None,
                'camera_unique_id': camera_unique_id,
                'camera_records_count': len(camera_id_list)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting detection stats by unique_id {camera_unique_id}: {e}")
            return {
                'total_detections': 0,
                'recent_detections_today': 0,
                'latest_detection_time': None,
                'camera_unique_id': camera_unique_id,
                'camera_records_count': 0
            }
        finally:
            session.close()
    
    def get_hourly_detection_stats(self, hours_back: int = 24, start_date: datetime = None, end_date: datetime = None) -> Dict[str, Any]:
        """Get hourly detection statistics for the last N hours OR a specific date range, grouped by camera"""
        session = self.get_session()
        try:
            # Determine time range - either from date range or hours_back
            if start_date and end_date:
                # Use specific date range
                range_start = start_date
                range_end = end_date
                use_date_range = True
            else:
                # Use hours_back (default behavior)
                range_start = get_ist_now() - timedelta(hours=hours_back)
                range_end = get_ist_now()
                use_date_range = False
            
            # Get all detections within the time range with camera info
            query = session.query(
                Detection.created_at,
                Detection.camera_id,
                Camera.name.label('camera_name')
            ).join(Camera).filter(
                Detection.created_at >= range_start,
                Detection.created_at <= range_end
            ).order_by(Detection.created_at.desc())
            
            detections = query.all()
            
            # Process data to group by hour
            hourly_data = {}
            camera_names = set()
            
            for detection in detections:
                # Round down to the hour
                hour_key = detection.created_at.replace(minute=0, second=0, microsecond=0)
                hour_str = hour_key.strftime('%Y-%m-%d %H:00')
                
                camera_name = detection.camera_name
                camera_names.add(camera_name)
                
                if hour_str not in hourly_data:
                    hourly_data[hour_str] = {}
                
                if camera_name not in hourly_data[hour_str]:
                    hourly_data[hour_str][camera_name] = 0
                
                hourly_data[hour_str][camera_name] += 1
            
            # Create complete hour range
            hours_list = []
            
            if use_date_range:
                # For date range, create hourly buckets between start and end
                current_hour = range_start.replace(minute=0, second=0, microsecond=0)
                end_hour = range_end.replace(minute=0, second=0, microsecond=0)
                
                while current_hour <= end_hour:
                    hours_list.append(current_hour.strftime('%Y-%m-%d %H:00'))
                    current_hour += timedelta(hours=1)
            else:
                # For hours_back, use existing logic
                current_time = get_ist_now().replace(minute=0, second=0, microsecond=0)
                
                for i in range(hours_back):
                    hour_time = current_time - timedelta(hours=i)
                    hours_list.append(hour_time.strftime('%Y-%m-%d %H:00'))
                
                hours_list.reverse()  # Chronological order
            
            # Prepare response data
            camera_names = sorted(list(camera_names))
            cameras_data = {}
            total_data = []
            
            for camera_name in camera_names:
                cameras_data[camera_name] = []
            
            for hour_str in hours_list:
                hour_total = 0
                
                for camera_name in camera_names:
                    count = hourly_data.get(hour_str, {}).get(camera_name, 0)
                    cameras_data[camera_name].append(count)
                    hour_total += count
                
                total_data.append(hour_total)
            
            return {
                'hours': hours_list,
                'total': total_data,
                'cameras': cameras_data
            }
            
        except Exception as e:
            logger.error(f"Error getting hourly detection stats: {e}")
            return {
                'hours': [],
                'total': [],
                'cameras': {}
            }
        finally:
            session.close()

    def cleanup_old_detections(self, days_to_keep: int = 30):
        """Clean up old detection records (optional maintenance)"""
        session = self.get_session()
        try:
            cutoff_date = get_ist_now().replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
            cutoff_date = cutoff_date - timedelta(days=days_to_keep)
            
            old_detections = session.query(Detection).filter(
                Detection.created_at < cutoff_date
            ).count()
            
            if old_detections > 0:
                session.query(Detection).filter(
                    Detection.created_at < cutoff_date
                ).delete()
                session.commit()
                
                logger.info(f"Cleaned up {old_detections} old detection records")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error cleaning up old detections: {e}")
        finally:
            session.close()

# Global database manager instance
db_manager = DatabaseManager()