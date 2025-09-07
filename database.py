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

Base = declarative_base()

class Camera(Base):
    """Camera metadata table"""
    __tablename__ = 'cameras'
    
    id = Column(Integer, primary_key=True, index=True)
    unique_id = Column(String(50), unique=True, index=True)
    name = Column(String(200), nullable=False)
    rtsp_url = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationship to detections
    detections = relationship("Detection", back_populates="camera")

class Detection(Base):
    """Person detection records table"""
    __tablename__ = 'detections'
    
    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(String(50), index=True)  # UUID for unique person identification
    camera_id = Column(Integer, ForeignKey('cameras.id'), nullable=False)
    camera_unique_id = Column(String(50), nullable=False)
    camera_name = Column(String(200), nullable=False)
    
    # Detection details
    confidence = Column(Float, nullable=False)
    bbox_x1 = Column(Float, nullable=False)
    bbox_y1 = Column(Float, nullable=False)
    bbox_x2 = Column(Float, nullable=False)
    bbox_y2 = Column(Float, nullable=False)
    
    # File paths
    full_image_path = Column(Text, nullable=False)  # Path to full frame image
    person_image_path = Column(Text)  # Path to cropped person image
    
    # Timestamps
    detected_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
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
            'camera_unique_id': self.camera_unique_id,
            'camera_name': self.camera_name,
            'confidence': round(self.confidence, 3),
            'bbox': [self.bbox_x1, self.bbox_y1, self.bbox_x2, self.bbox_y2],
            'full_image_path': self.full_image_path,
            'person_image_path': self.person_image_path,
            'detected_at': self.detected_at.isoformat(),
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
                    camera_unique_id = f"camera_{camera_id}_{int(datetime.utcnow().timestamp())}"
                
                camera = Camera(
                    id=camera_id,
                    unique_id=camera_unique_id,
                    name=camera_name,
                    rtsp_url=rtsp_url,
                    created_at=datetime.utcnow(),
                    is_active=True
                )
                session.add(camera)
                session.commit()
                logger.info(f"Created new camera record: {camera_name} (ID: {camera_id})")
            else:
                # Update camera info if needed
                camera.name = camera_name
                camera.is_active = True
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
    
    def save_detection(self, camera_id: int, camera_name: str, 
                      detection_data: Dict[str, Any]) -> Detection:
        """Save person detection to database"""
        session = self.get_session()
        try:
            # Handle camera creation/retrieval within THIS session to avoid detached instances
            camera = session.query(Camera).filter(Camera.id == camera_id).first()
            
            if not camera:
                # Create new camera record in the same session
                camera_unique_id = detection_data.get('camera_unique_id')
                if not camera_unique_id:
                    camera_unique_id = f"camera_{camera_id}_{int(datetime.utcnow().timestamp())}"
                
                camera = Camera(
                    id=camera_id,
                    unique_id=camera_unique_id,
                    name=camera_name,
                    rtsp_url=detection_data.get('rtsp_url', ''),
                    created_at=datetime.utcnow(),
                    is_active=True
                )
                session.add(camera)
                session.flush()  # Ensure camera gets ID without committing yet
                logger.info(f"Created new camera record: {camera_name} (ID: {camera_id})")
            else:
                # Update existing camera info in the same session
                camera.name = camera_name
                camera.is_active = True
                logger.debug(f"Updated camera record: {camera_name} (ID: {camera_id})")
            
            # Generate unique person ID
            person_id = str(uuid.uuid4())
            
            # Create detection record using the attached camera object
            detection = Detection(
                person_id=person_id,
                camera_id=camera_id,
                camera_unique_id=camera.unique_id,  # Now safe to access - camera is attached
                camera_name=camera_name,
                confidence=detection_data['confidence'],
                bbox_x1=detection_data['bbox'][0],
                bbox_y1=detection_data['bbox'][1],
                bbox_x2=detection_data['bbox'][2],
                bbox_y2=detection_data['bbox'][3],
                full_image_path=detection_data['full_image_path'],
                person_image_path=detection_data.get('person_image_path'),
                detected_at=detection_data['timestamp'],
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
            
            logger.info(f"Saved detection: Person {person_id} in camera {camera_name}")
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
                query = query.filter(Detection.detected_at >= start_date)
            
            if end_date:
                query = query.filter(Detection.detected_at <= end_date)
            
            # Order by most recent first and limit results
            detections = query.order_by(Detection.detected_at.desc()).limit(limit).all()
            
            return detections
            
        except Exception as e:
            logger.error(f"Error getting detection history: {e}")
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
            recent_time = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            recent_detections = query.filter(Detection.detected_at >= recent_time).count()
            
            # Get latest detection
            latest_detection = query.order_by(Detection.detected_at.desc()).first()
            
            stats = {
                'total_detections': total_detections,
                'recent_detections_today': recent_detections,
                'latest_detection_time': latest_detection.detected_at.isoformat() if latest_detection else None,
                'camera_id': camera_id
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting detection stats: {e}")
            return {'total_detections': 0, 'recent_detections_today': 0}
        finally:
            session.close()
    
    def cleanup_old_detections(self, days_to_keep: int = 30):
        """Clean up old detection records (optional maintenance)"""
        session = self.get_session()
        try:
            cutoff_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            cutoff_date = cutoff_date - timedelta(days=days_to_keep)
            
            old_detections = session.query(Detection).filter(
                Detection.detected_at < cutoff_date
            ).count()
            
            if old_detections > 0:
                session.query(Detection).filter(
                    Detection.detected_at < cutoff_date
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