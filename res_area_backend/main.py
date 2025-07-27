import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Set
import uuid
import cv2
import numpy as np
import json
from pathlib import Path
from contextlib import asynccontextmanager
import threading
import time

from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseSettings, BaseModel
import motor.motor_asyncio
from bson import ObjectId
import asyncio
import uvicorn
import yt_dlp

# Configuration
class Settings(BaseSettings):
    # Database
    mongodb_url: str = "mongodb://localhost:27017"
    database_name: str = "monitoring_system"
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    
    # File uploads
    upload_path: str = "uploads"
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_video_extensions: List[str] = [".mp4", ".avi", ".mov", ".mkv"]
    
    # AI Model
    model_confidence_threshold: float = 0.5
    yolo_config_path: str = "models/yolov4.cfg"
    yolo_weights_path: str = "models/yolov4.weights"
    coco_names_path: str = "models/coco.names"
    
    class Config:
        env_file = ".env"

settings = Settings()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Database Models
class Camera(BaseModel):
    id: Optional[str] = None
    name: str
    device_id: str
    status: str = "active"
    created_at: Optional[datetime] = None

class CameraUpdate(BaseModel):
    name: Optional[str] = None
    device_id: Optional[str] = None
    status: Optional[str] = None

class Detection(BaseModel):
    id: Optional[str] = None
    camera_id: str
    timestamp: datetime
    confidence: float
    image_path: Optional[str] = None
    person_count: int = 1
    bounding_boxes: List[Dict] = []

class SystemLog(BaseModel):
    id: Optional[str] = None
    event_type: str
    message: str
    timestamp: Optional[datetime] = None
    level: str = "info"

class SystemStatus(BaseModel):
    uptime: str
    active_cameras: int
    total_detections: int
    system_health: str
    last_detection: Optional[datetime] = None
    mongodb_status: str

class VideoProcessRequest(BaseModel):
    youtube_url: str

# MongoDB Connection Manager
class DatabaseManager:
    def __init__(self):
        self.client = None
        self.db = None
        self.cameras_collection = None
        self.detections_collection = None
        self.logs_collection = None
        
    async def connect(self):
        """Connect to MongoDB"""
        try:
            self.client = motor.motor_asyncio.AsyncIOMotorClient(settings.mongodb_url)
            self.db = self.client[settings.database_name]
            
            # Initialize collections
            self.cameras_collection = self.db.cameras
            self.detections_collection = self.db.detections
            self.logs_collection = self.db.system_logs
            
            # Test connection
            await self.client.admin.command('ping')
            
            # Create indexes for better performance
            await self.create_indexes()
            
            logger.info("Connected to MongoDB successfully")
            await self.log_system_event("startup", "Database connection established")
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    async def create_indexes(self):
        """Create database indexes for better performance"""
        try:
            # Cameras indexes
            await self.cameras_collection.create_index("device_id", unique=True)
            await self.cameras_collection.create_index("status")
            
            # Detections indexes
            await self.detections_collection.create_index("camera_id")
            await self.detections_collection.create_index("timestamp")
            await self.detections_collection.create_index([("camera_id", 1), ("timestamp", -1)])
            
            # Logs indexes
            await self.logs_collection.create_index("timestamp")
            await self.logs_collection.create_index("event_type")
            
            logger.info("Database indexes created successfully")
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
    
    async def log_system_event(self, event_type: str, message: str, level: str = "info"):
        """Log system events to MongoDB"""
        try:
            log_entry = {
                "_id": str(ObjectId()),
                "event_type": event_type,
                "message": message,
                "level": level,
                "timestamp": datetime.utcnow()
            }
            await self.logs_collection.insert_one(log_entry)
        except Exception as e:
            logger.error(f"Failed to log system event: {e}")
    
    async def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")

# AI Model Manager
class PersonDetectionModel:
    def __init__(self):
        self.net = None
        self.output_layers = None
        self.classes = None
        self.model_loaded = False
        self.load_model()
    
    def load_model(self):
        """Load YOLO model for person detection"""
        try:
            # Load YOLO
            if os.path.exists(settings.yolo_weights_path) and os.path.exists(settings.yolo_config_path):
                self.net = cv2.dnn.readNet(settings.yolo_weights_path, settings.yolo_config_path)
                layer_names = self.net.getLayerNames()
                self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
                
                # Load class names
                if os.path.exists(settings.coco_names_path):
                    with open(settings.coco_names_path, "r") as f:
                        self.classes = [line.strip() for line in f.readlines()]
                
                self.model_loaded = True
                logger.info("YOLO model loaded successfully")
            else:
                logger.warning("YOLO model files not found, using basic detection")
                
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
    
    def detect_persons(self, image):
        """Detect persons in image"""
        try:
            if not self.model_loaded or self.net is None:
                # Fallback to basic detection using OpenCV
                return self._basic_person_detection(image)
            
            height, width, channels = image.shape
            
            # Detecting objects
            blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.output_layers)
            
            # Information to show on screen
            class_ids = []
            confidences = []
            boxes = []
            
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    # Only detect persons (class_id = 0 in COCO dataset)
                    if class_id == 0 and confidence > settings.model_confidence_threshold:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # Non-max suppression
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            
            detections = []
            if len(indexes) > 0:
                for i in indexes.flatten():
                    detections.append({
                        'confidence': confidences[i],
                        'box': boxes[i]
                    })
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in person detection: {e}")
            return []
    
    def _basic_person_detection(self, image):
        """Basic person detection using OpenCV HOG"""
        try:
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            
            boxes, weights = hog.detectMultiScale(image, winStride=(8,8), padding=(32,32), scale=1.05)
            
            detections = []
            for i, (x, y, w, h) in enumerate(boxes):
                if weights[i] > 0.5:
                    detections.append({
                        'confidence': float(weights[i]),
                        'box': [x, y, w, h]
                    })
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in basic person detection: {e}")
            return []

# WebSocket Manager
class WebSocketManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.camera_feeds: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, camera_id: str = None):
        await websocket.accept()
        self.active_connections.add(websocket)
        if camera_id:
            self.camera_feeds[camera_id] = websocket
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket, camera_id: str = None):
        self.active_connections.discard(websocket)
        if camera_id and camera_id in self.camera_feeds:
            del self.camera_feeds[camera_id]
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast_alert(self, message: dict):
        """Broadcast alert to all connected clients"""
        if self.active_connections:
            disconnected = set()
            for connection in self.active_connections:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Failed to send message to WebSocket: {e}")
                    disconnected.add(connection)
            
            # Remove disconnected connections
            for conn in disconnected:
                self.active_connections.discard(conn)
    
    async def send_camera_frame(self, camera_id: str, frame_data: bytes):
        """Send camera frame to specific connection"""
        if camera_id in self.camera_feeds:
            try:
                await self.camera_feeds[camera_id].send_bytes(frame_data)
            except Exception as e:
                logger.error(f"Failed to send frame to camera {camera_id}: {e}")
                del self.camera_feeds[camera_id]

# Camera Manager
class CameraManager:
    def __init__(self, db_manager: DatabaseManager, ws_manager: WebSocketManager, detection_model: PersonDetectionModel):
        self.db_manager = db_manager
        self.ws_manager = ws_manager
        self.detection_model = detection_model
        self.active_cameras: Dict[str, cv2.VideoCapture] = {}
        self.processing_threads: Dict[str, threading.Thread] = {}
        self.stop_flags: Dict[str, threading.Event] = {}
    
    async def add_camera(self, camera: Camera) -> str:
        """Add a new camera to the system"""
        try:
            camera_id = str(ObjectId())
            camera_doc = {
                "_id": camera_id,
                "name": camera.name,
                "device_id": camera.device_id,
                "status": camera.status,
                "created_at": datetime.utcnow()
            }
            
            await self.db_manager.cameras_collection.insert_one(camera_doc)
            await self.db_manager.log_system_event("camera_added", f"Camera {camera.name} added successfully")
            
            # Start camera processing if active
            if camera.status == "active":
                await self.start_camera_processing(camera_id, camera.device_id)
            
            return camera_id
            
        except Exception as e:
            logger.error(f"Failed to add camera: {e}")
            await self.db_manager.log_system_event("camera_error", f"Failed to add camera: {str(e)}", "error")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def start_camera_processing(self, camera_id: str, device_id: str):
        """Start processing camera feed"""
        try:
            # Try to parse device_id as integer for camera index, otherwise use as string for IP camera
            try:
                device = int(device_id)
            except ValueError:
                device = device_id
            
            cap = cv2.VideoCapture(device)
            if not cap.isOpened():
                raise Exception(f"Cannot open camera {device_id}")
            
            self.active_cameras[camera_id] = cap
            self.stop_flags[camera_id] = threading.Event()
            
            # Start processing thread
            thread = threading.Thread(target=self._process_camera_feed, args=(camera_id,))
            thread.daemon = True
            thread.start()
            self.processing_threads[camera_id] = thread
            
            logger.info(f"Started processing camera {camera_id}")
            
        except Exception as e:
            logger.error(f"Failed to start camera processing: {e}")
            await self.db_manager.log_system_event("camera_error", f"Failed to start camera {camera_id}: {str(e)}", "error")
    
    def _process_camera_feed(self, camera_id: str):
        """Process camera feed in separate thread"""
        cap = self.active_cameras.get(camera_id)
        stop_flag = self.stop_flags.get(camera_id)
        
        if not cap or not stop_flag:
            return
        
        frame_count = 0
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        while not stop_flag.is_set():
            try:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process every 30th frame for person detection (reduce CPU load)
                if frame_count % 30 == 0:
                    detections = self.detection_model.detect_persons(frame)
                    
                    if detections:
                        # Save detection to database
                        loop.run_until_complete(self._save_detection(camera_id, detections, frame))
                        
                        # Send alert via WebSocket
                        alert_data = {
                            "type": "person_detected",
                            "camera_id": camera_id,
                            "timestamp": datetime.utcnow().isoformat(),
                            "person_count": len(detections),
                            "confidence": max([d['confidence'] for d in detections])
                        }
                        loop.run_until_complete(self.ws_manager.broadcast_alert(alert_data))
                
                # Send frame via WebSocket (every 5th frame to reduce bandwidth)
                if frame_count % 5 == 0:
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_data = buffer.tobytes()
                    loop.run_until_complete(self.ws_manager.send_camera_frame(camera_id, frame_data))
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Error processing camera {camera_id}: {e}")
                break
        
        # Cleanup
        cap.release()
        if camera_id in self.active_cameras:
            del self.active_cameras[camera_id]
        loop.close()
    
    async def _save_detection(self, camera_id: str, detections: List[Dict], frame):
        """Save detection to database"""
        try:
            # Save frame image
            timestamp = datetime.utcnow()
            image_filename = f"detection_{camera_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
            image_path = os.path.join(settings.upload_path, "detections", image_filename)
            
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            cv2.imwrite(image_path, frame)
            
            detection_doc = {
                "_id": str(ObjectId()),
                "camera_id": camera_id,
                "timestamp": timestamp,
                "confidence": max([d['confidence'] for d in detections]),
                "person_count": len(detections),
                "bounding_boxes": [d['box'] for d in detections],
                "image_path": image_path
            }
            
            await self.db_manager.detections_collection.insert_one(detection_doc)
            await self.db_manager.log_system_event("detection", f"Person detected on camera {camera_id}")
            
        except Exception as e:
            logger.error(f"Failed to save detection: {e}")
    
    async def stop_camera_processing(self, camera_id: str):
        """Stop processing camera feed"""
        try:
            if camera_id in self.stop_flags:
                self.stop_flags[camera_id].set()
            
            if camera_id in self.processing_threads:
                self.processing_threads[camera_id].join(timeout=5)
                del self.processing_threads[camera_id]
            
            if camera_id in self.active_cameras:
                self.active_cameras[camera_id].release()
                del self.active_cameras[camera_id]
            
            if camera_id in self.stop_flags:
                del self.stop_flags[camera_id]
            
            logger.info(f"Stopped processing camera {camera_id}")
            
        except Exception as e:
            logger.error(f"Failed to stop camera processing: {e}")

# Video Processing Manager
class VideoProcessor:
    def __init__(self, detection_model: PersonDetectionModel, db_manager: DatabaseManager):
        self.detection_model = detection_model
        self.db_manager = db_manager
    
    async def process_uploaded_video(self, file_path: str) -> Dict:
        """Process uploaded video file for person detection"""
        try:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                raise Exception("Cannot open video file")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            detections = []
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every 30th frame to reduce processing time
                if frame_count % 30 == 0:
                    frame_detections = self.detection_model.detect_persons(frame)
                    if frame_detections:
                        timestamp = frame_count / fps if fps > 0 else frame_count
                        detections.append({
                            "timestamp": timestamp,
                            "person_count": len(frame_detections),
                            "confidence": max([d['confidence'] for d in frame_detections]),
                            "frame_number": frame_count
                        })
                
                frame_count += 1
            
            cap.release()
            
            result = {
                "total_frames": total_frames,
                "duration": duration,
                "fps": fps,
                "detections": detections,
                "total_detections": len(detections),
                "processed_at": datetime.utcnow().isoformat()
            }
            
            await self.db_manager.log_system_event("video_processed", f"Video processed: {len(detections)} detections found")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process video: {e}")
            await self.db_manager.log_system_event("video_error", f"Failed to process video: {str(e)}", "error")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def process_youtube_video(self, youtube_url: str) -> Dict:
        """Download and process YouTube video"""
        try:
            # Download YouTube video
            ydl_opts = {
                'format': 'mp4[height<=720]',
                'outtmpl': os.path.join(settings.upload_path, 'youtube', '%(title)s.%(ext)s'),
                'noplaylist': True,
            }
            
            os.makedirs(os.path.join(settings.upload_path, 'youtube'), exist_ok=True)
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=True)
                video_title = info.get('title', 'Unknown')
                video_path = ydl.prepare_filename(info)
            
            # Process the downloaded video
            result = await self.process_uploaded_video(video_path)
            result['source'] = 'youtube'
            result['video_title'] = video_title
            result['original_url'] = youtube_url
            
            # Clean up downloaded file
            if os.path.exists(video_path):
                os.remove(video_path)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process YouTube video: {e}")
            await self.db_manager.log_system_event("youtube_error", f"Failed to process YouTube video: {str(e)}", "error")
            raise HTTPException(status_code=500, detail=str(e))

# Global variables
db_manager = DatabaseManager()
detection_model = PersonDetectionModel()
ws_manager = WebSocketManager()
camera_manager = None
video_processor = None
app_start_time = datetime.utcnow()

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global camera_manager, video_processor
    
    # Create upload directories
    os.makedirs(settings.upload_path, exist_ok=True)
    os.makedirs(os.path.join(settings.upload_path, "detections"), exist_ok=True)
    os.makedirs(os.path.join(settings.upload_path, "videos"), exist_ok=True)
    
    # Initialize database
    await db_manager.connect()
    
    # Initialize managers
    camera_manager = CameraManager(db_manager, ws_manager, detection_model)
    video_processor = VideoProcessor(detection_model, db_manager)
    
    # Load existing active cameras
    async for camera_doc in db_manager.cameras_collection.find({"status": "active"}):
        try:
            await camera_manager.start_camera_processing(camera_doc["_id"], camera_doc["device_id"])
        except Exception as e:
            logger.error(f"Failed to start camera {camera_doc['_id']}: {e}")
    
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    # Stop all camera processing
    if camera_manager:
        for camera_id in list(camera_manager.active_cameras.keys()):
            await camera_manager.stop_camera_processing(camera_id)
    
    # Close database connection
    await db_manager.close()
    logger.info("Application shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Restricted Area Monitoring System",
    description="AI-powered person detection and monitoring system",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Endpoints

@app.get("/api/cameras")
async def get_cameras():
    """Get all cameras"""
    try:
        cameras = []
        async for camera_doc in db_manager.cameras_collection.find():
            camera_doc["id"] = camera_doc.pop("_id")
            cameras.append(camera_doc)
        
        return {"cameras": cameras, "total": len(cameras)}
    
    except Exception as e:
        logger.error(f"Failed to get cameras: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/cameras")
async def add_camera(camera: Camera):
    """Add new camera"""
    try:
        # Check if device_id already exists
        existing = await db_manager.cameras_collection.find_one({"device_id": camera.device_id})
        if existing:
            raise HTTPException(status_code=400, detail="Camera with this device_id already exists")
        
        camera_id = await camera_manager.add_camera(camera)
        
        return {
            "id": camera_id,
            "message": "Camera added successfully",
            "status": "success"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add camera: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/cameras/{camera_id}")
async def remove_camera(camera_id: str):
    """Remove camera"""
    try:
        # Stop camera processing
        await camera_manager.stop_camera_processing(camera_id)
        
        # Delete from database
        result = await db_manager.cameras_collection.delete_one({"_id": camera_id})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Camera not found")
        
        await db_manager.log_system_event("camera_removed", f"Camera {camera_id} removed")
        
        return {"message": "Camera removed successfully", "status": "success"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove camera: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/cameras/{camera_id}")
async def update_camera(camera_id: str, camera_update: CameraUpdate):
    """Update camera settings"""
    try:
        update_data = {k: v for k, v in camera_update.dict().items() if v is not None}
        
        if not update_data:
            raise HTTPException(status_code=400, detail="No valid fields to update")
        
        result = await db_manager.cameras_collection.update_one(
            {"_id": camera_id},
            {"$set": update_data}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Camera not found")
        
        # Handle status change
        if "status" in update_data:
            if update_data["status"] == "active":
                camera_doc = await db_manager.cameras_collection.find_one({"_id": camera_id})
                if camera_doc:
                    await camera_manager.start_camera_processing(camera_id, camera_doc["device_id"])
            else:
                await camera_manager.stop_camera_processing(camera_id)
        
        await db_manager.log_system_event("camera_updated", f"Camera {camera_id} updated")
        
        return {"message": "Camera updated successfully", "status": "success"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update camera: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/detections")
async def get_detections(
    camera_id: Optional[str] = None,
    limit: int = 100,
    skip: int = 0,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Get detection history"""
    try:
        query = {}
        
        if camera_id:
            query["camera_id"] = camera_id
        
        if start_date or end_date:
            date_query = {}
            if start_date:
                date_query["$gte"] = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            if end_date:
                date_query["$lte"] = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            query["timestamp"] = date_query
        
        # Get total count
        total = await db_manager.detections_collection.count_documents(query)
        
        # Get detections
        cursor = db_manager.detections_collection.find(query).sort("timestamp", -1).skip(skip).limit(limit)
        detections = []
        
        async for detection_doc in cursor:
            detection_doc["id"] = detection_doc.pop("_id")
            detections.append(detection_doc)
        
        return {
            "detections": detections,
            "total": total,
            "limit": limit,
            "skip": skip
        }
    
    except Exception as e:
        logger.error(f"Failed to get detections: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Continuation from main.py - Upload Video and remaining endpoints

@app.post("/api/upload-video")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload and process video file"""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in settings.allowed_video_extensions:
            raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {settings.allowed_video_extensions}")
        
        # Check file size
        file_size = 0
        content = await file.read()
        file_size = len(content)
        
        if file_size > settings.max_file_size:
            raise HTTPException(status_code=400, detail=f"File too large. Max size: {settings.max_file_size} bytes")
        
        # Save uploaded file
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename}"
        file_path = os.path.join(settings.upload_path, "videos", filename)
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Process video in background
        background_tasks.add_task(process_video_background, file_path, file_id)
        
        await db_manager.log_system_event("video_upload", f"Video uploaded: {file.filename}")
        
        return {
            "message": "Video uploaded successfully and queued for processing",
            "file_id": file_id,
            "filename": filename,
            "status": "processing"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload video: {e}")
        await db_manager.log_system_event("video_error", f"Failed to upload video: {str(e)}", "error")
        raise HTTPException(status_code=500, detail=str(e))

async def process_video_background(file_path: str, file_id: str):
    """Background task to process uploaded video"""
    try:
        result = await video_processor.process_uploaded_video(file_path)
        
        # Store processing result in database
        processing_result = {
            "_id": file_id,
            "file_path": file_path,
            "processing_result": result,
            "status": "completed",
            "processed_at": datetime.utcnow()
        }
        
        await db_manager.db.video_processing.insert_one(processing_result)
        
        # Send notification via WebSocket
        await ws_manager.broadcast_alert({
            "type": "video_processed",
            "file_id": file_id,
            "detections_count": result["total_detections"],
            "status": "completed"
        })
        
        logger.info(f"Video processing completed for file_id: {file_id}")
        
    except Exception as e:
        logger.error(f"Background video processing failed: {e}")
        
        # Update status to failed
        processing_result = {
            "_id": file_id,
            "file_path": file_path,
            "status": "failed",
            "error": str(e),
            "processed_at": datetime.utcnow()
        }
        
        try:
            await db_manager.db.video_processing.insert_one(processing_result)
        except:
            pass

@app.post("/api/process-youtube")
async def process_youtube(request: VideoProcessRequest, background_tasks: BackgroundTasks):
    """Process YouTube video URL"""
    try:
        # Validate YouTube URL
        if not request.youtube_url:
            raise HTTPException(status_code=400, detail="YouTube URL is required")
        
        if "youtube.com" not in request.youtube_url and "youtu.be" not in request.youtube_url:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL")
        
        # Generate unique ID for this processing task
        task_id = str(uuid.uuid4())
        
        # Process YouTube video in background
        background_tasks.add_task(process_youtube_background, request.youtube_url, task_id)
        
        await db_manager.log_system_event("youtube_processing", f"YouTube processing started: {request.youtube_url}")
        
        return {
            "message": "YouTube video queued for processing",
            "task_id": task_id,
            "url": request.youtube_url,
            "status": "processing"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to queue YouTube processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_youtube_background(youtube_url: str, task_id: str):
    """Background task to process YouTube video"""
    try:
        result = await video_processor.process_youtube_video(youtube_url)
        
        # Store processing result
        processing_result = {
            "_id": task_id,
            "youtube_url": youtube_url,
            "processing_result": result,
            "status": "completed",
            "processed_at": datetime.utcnow()
        }
        
        await db_manager.db.youtube_processing.insert_one(processing_result)
        
        # Send notification via WebSocket
        await ws_manager.broadcast_alert({
            "type": "youtube_processed",
            "task_id": task_id,
            "video_title": result.get("video_title", "Unknown"),
            "detections_count": result["total_detections"],
            "status": "completed"
        })
        
        logger.info(f"YouTube processing completed for task_id: {task_id}")
        
    except Exception as e:
        logger.error(f"Background YouTube processing failed: {e}")
        
        # Update status to failed
        processing_result = {
            "_id": task_id,
            "youtube_url": youtube_url,
            "status": "failed",
            "error": str(e),
            "processed_at": datetime.utcnow()
        }
        
        try:
            await db_manager.db.youtube_processing.insert_one(processing_result)
        except:
            pass
        
        # Send error notification
        try:
            await ws_manager.broadcast_alert({
                "type": "youtube_processing_error",
                "task_id": task_id,
                "error": str(e),
                "status": "failed"
            })
        except:
            pass

@app.get("/api/processing-status/{task_id}")
async def get_processing_status(task_id: str):
    """Get processing status for video or YouTube task"""
    try:
        # Check video processing collection
        video_result = await db_manager.db.video_processing.find_one({"_id": task_id})
        if video_result:
            return {
                "task_id": task_id,
                "type": "video",
                "status": video_result.get("status"),
                "result": video_result.get("processing_result"),
                "error": video_result.get("error"),
                "processed_at": video_result.get("processed_at")
            }
        
        # Check YouTube processing collection
        youtube_result = await db_manager.db.youtube_processing.find_one({"_id": task_id})
        if youtube_result:
            return {
                "task_id": task_id,
                "type": "youtube",
                "status": youtube_result.get("status"),
                "result": youtube_result.get("processing_result"),
                "error": youtube_result.get("error"),
                "processed_at": youtube_result.get("processed_at")
            }
        
        raise HTTPException(status_code=404, detail="Processing task not found")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get processing status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/system-status")
async def get_system_status():
    """Get system status and statistics"""
    try:
        # Calculate uptime
        uptime_delta = datetime.utcnow() - app_start_time
        uptime_str = str(uptime_delta).split('.')[0]  # Remove microseconds
        
        # Get active cameras count
        active_cameras = await db_manager.cameras_collection.count_documents({"status": "active"})
        
        # Get total detections count
        total_detections = await db_manager.detections_collection.count_documents({})
        
        # Get last detection
        last_detection_doc = await db_manager.detections_collection.find_one(
            {},
            sort=[("timestamp", -1)]
        )
        last_detection = last_detection_doc.get("timestamp") if last_detection_doc else None
        
        # Check MongoDB status
        try:
            await db_manager.client.admin.command('ping')
            mongodb_status = "connected"
        except:
            mongodb_status = "disconnected"
        
        # Determine system health
        system_health = "healthy"
        if mongodb_status == "disconnected":
            system_health = "unhealthy"
        elif active_cameras == 0:
            system_health = "warning"
        
        # Get recent system logs
        recent_logs = []
        async for log_doc in db_manager.logs_collection.find().sort("timestamp", -1).limit(5):
            log_doc["id"] = log_doc.pop("_id")
            recent_logs.append(log_doc)
        
        # Get detection statistics for last 24 hours
        yesterday = datetime.utcnow() - timedelta(days=1)
        detections_24h = await db_manager.detections_collection.count_documents({
            "timestamp": {"$gte": yesterday}
        })
        
        status = SystemStatus(
            uptime=uptime_str,
            active_cameras=active_cameras,
            total_detections=total_detections,
            system_health=system_health,
            last_detection=last_detection,
            mongodb_status=mongodb_status
        )
        
        return {
            **status.dict(),
            "detections_24h": detections_24h,
            "recent_logs": recent_logs,
            "active_camera_feeds": len(camera_manager.active_cameras) if camera_manager else 0,
            "websocket_connections": len(ws_manager.active_connections),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/system-logs")
async def get_system_logs(
    limit: int = 100,
    skip: int = 0,
    level: Optional[str] = None,
    event_type: Optional[str] = None
):
    """Get system logs"""
    try:
        query = {}
        
        if level:
            query["level"] = level
        if event_type:
            query["event_type"] = event_type
        
        # Get total count
        total = await db_manager.logs_collection.count_documents(query)
        
        # Get logs
        cursor = db_manager.logs_collection.find(query).sort("timestamp", -1).skip(skip).limit(limit)
        logs = []
        
        async for log_doc in cursor:
            log_doc["id"] = log_doc.pop("_id")
            logs.append(log_doc)
        
        return {
            "logs": logs,
            "total": total,
            "limit": limit,
            "skip": skip
        }
    
    except Exception as e:
        logger.error(f"Failed to get system logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/system-logs")
async def clear_system_logs(older_than_days: int = 30):
    """Clear old system logs"""
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)
        
        result = await db_manager.logs_collection.delete_many({
            "timestamp": {"$lt": cutoff_date}
        })
        
        await db_manager.log_system_event("logs_cleared", f"Cleared {result.deleted_count} old log entries")
        
        return {
            "message": f"Cleared {result.deleted_count} log entries older than {older_than_days} days",
            "deleted_count": result.deleted_count,
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Failed to clear system logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/statistics")
async def get_statistics():
    """Get detection statistics"""
    try:
        # Daily detection counts for last 7 days
        daily_stats = []
        for i in range(7):
            date = datetime.utcnow() - timedelta(days=i)
            start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_day = start_of_day + timedelta(days=1)
            
            count = await db_manager.detections_collection.count_documents({
                "timestamp": {"$gte": start_of_day, "$lt": end_of_day}
            })
            
            daily_stats.append({
                "date": start_of_day.isoformat()[:10],
                "detections": count
            })
        
        # Camera-wise detection counts
        camera_stats = []
        async for camera_doc in db_manager.cameras_collection.find():
            camera_id = camera_doc["_id"]
            detection_count = await db_manager.detections_collection.count_documents({
                "camera_id": camera_id
            })
            
            camera_stats.append({
                "camera_id": camera_id,
                "camera_name": camera_doc["name"],
                "detections": detection_count
            })
        
        # Hourly distribution for today
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        hourly_stats = []
        
        for hour in range(24):
            hour_start = today + timedelta(hours=hour)
            hour_end = hour_start + timedelta(hours=1)
            
            count = await db_manager.detections_collection.count_documents({
                "timestamp": {"$gte": hour_start, "$lt": hour_end}
            })
            
            hourly_stats.append({
                "hour": hour,
                "detections": count
            })
        
        return {
            "daily_stats": list(reversed(daily_stats)),  # Reverse to show oldest first
            "camera_stats": camera_stats,
            "hourly_stats": hourly_stats,
            "generated_at": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await ws_manager.connect(websocket)
    
    try:
        while True:
            # Keep connection alive and handle any incoming messages
            data = await websocket.receive_text()
            
            # Parse incoming message
            try:
                message = json.loads(data)
                message_type = message.get("type")
                
                if message_type == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": datetime.utcnow().isoformat()})
                elif message_type == "subscribe_camera":
                    camera_id = message.get("camera_id")
                    if camera_id:
                        ws_manager.camera_feeds[camera_id] = websocket
                elif message_type == "unsubscribe_camera":
                    camera_id = message.get("camera_id")
                    if camera_id and camera_id in ws_manager.camera_feeds:
                        del ws_manager.camera_feeds[camera_id]
                        
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
            
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        ws_manager.disconnect(websocket)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        await db_manager.client.admin.command('ping')
        db_status = "healthy"
    except:
        db_status = "unhealthy"
    
    return {
        "status": "healthy" if db_status == "healthy" else "unhealthy",
        "database": db_status,
        "timestamp": datetime.utcnow().isoformat(),
        "uptime": str(datetime.utcnow() - app_start_time).split('.')[0]
    }

# Static file serving for detection images
from fastapi.staticfiles import StaticFiles
app.mount("/uploads", StaticFiles(directory=settings.upload_path), name="uploads")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found", "status": "error"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "status": "error"}
    )

# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )

