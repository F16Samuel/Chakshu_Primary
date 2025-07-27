import os
import logging
import asyncio
import base64
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime, timedelta
import uuid # For generating unique video IDs

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from ultralytics import YOLO

# New import for async MongoDB driver
import motor.motor_asyncio
from pymongo.errors import ConnectionFailure

# Import the new modules
from models.chunker import chunk_file, get_file_md5
from models.collector import reassemble_file

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for response structure
class Detection(BaseModel):
    label: str
    confidence: float
    bbox: List[int]
    timestamp: str

class DetectionResponse(BaseModel):
    detections: List[Detection]
    frame_id: Optional[int] = None
    processing_time: float
    timestamp: str
    camera_id: str # Added camera_id
    camera_name: Optional[str] = None # Added camera_name

# New Pydantic model for chunking multiple files
class ChunkRequest(BaseModel):
    model_paths: List[str]
    chunk_size_mb: int = 80

# New Pydantic model for individual video frame detection logs
class VideoFrameDetection(BaseModel):
    frame_number: int
    timestamp: str
    detections: List[Detection]
    frame_image_base64: Optional[str] = None # Added field for base64 image of the frame

# New Pydantic model for overall video processing response
class VideoProcessResponse(BaseModel):
    video_id: str
    status: str
    total_frames_processed: int
    total_detections: int
    processing_duration_seconds: float
    detections_by_frame: List[VideoFrameDetection]
    message: str

class PerformanceStats:
    """Track performance statistics for monitoring."""
    
    def __init__(self):
        self.total_frames = 0
        self.total_detections = 0
        self.total_processing_time = 0.0
        self.start_time = datetime.now()
        logger.info("PerformanceStats initialized")
    
    def update(self, processing_time: float, detection_count: int):
        """Update performance statistics"""
        self.total_frames += 1
        self.total_detections += detection_count
        self.total_processing_time += processing_time
    
    def get_stats(self) -> Dict[str, float]:
        """Get current performance statistics"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        avg_fps = self.total_frames / self.total_processing_time if self.total_processing_time > 0 else 0
        
        return {
            "total_frames_processed": self.total_frames,
            "total_detections": self.total_detections,
            "average_fps": round(avg_fps, 2),
            "uptime_seconds": round(uptime, 1),
            "average_processing_time": round(self.total_processing_time / self.total_frames if self.total_frames > 0 else 0, 4)
        }

class DatabaseManager:
    """Manages MongoDB database operations for threat logs and video detections."""
    def __init__(self, mongo_uri: str, db_name: str):
        self.client = motor.motor_asyncio.AsyncIOMotorClient(mongo_uri)
        self.db = self.client[db_name]
        self.threat_logs_collection = self.db["threat_logs"]
        self.video_detections_collection = self.db["video_detections"]
        logger.info(f"DatabaseManager initialized with MongoDB: {db_name}")

    async def _check_connection(self):
        """Checks if the MongoDB connection is active."""
        try:
            # The ping command is cheap and does not require write permissions.
            await self.client.admin.command('ping')
            logger.info("MongoDB connection successful.")
            return True
        except ConnectionFailure as e:
            logger.error(f"MongoDB connection failed: {e}")
            return False

    async def _create_indexes(self):
        """Creates indexes for better query performance."""
        try:
            # For threat_logs: index on timestamp for sorting
            await self.threat_logs_collection.create_index("timestamp", background=True)
            await self.threat_logs_collection.create_index("camera_id", background=True)
            # For video_detections: index on video_id and frame_number for retrieval and sorting
            await self.video_detections_collection.create_index("video_id", background=True)
            await self.video_detections_collection.create_index([("frame_number", 1), ("detection_timestamp", 1)], background=True)
            logger.info("MongoDB indexes ensured.")
        except Exception as e:
            logger.error(f"Error creating MongoDB indexes: {e}")

    async def log_threat_event(self, camera_id: str, camera_name: str, action: str, confidence: Optional[float] = None):
        """
        Logs a threat event (entry or exit) to the threat_logs collection.
        """
        try:
            timestamp = datetime.now()
            log_entry = {
                "camera_id": camera_id,
                "camera_name": camera_name,
                "action": action,
                "timestamp": timestamp, # Stored as ISODate
                "method": "scanner",
                "confidence": confidence
            }
            await self.threat_logs_collection.insert_one(log_entry)
            logger.info(f"Logged event: Camera '{camera_name}' ({camera_id}) - Action: {action}, Confidence: {confidence}")
        except Exception as e:
            logger.error(f"Error logging threat event to MongoDB: {e}")

    async def log_video_detection(
        self,
        video_id: str,
        camera_id: str,
        camera_name: str,
        frame_number: int,
        detection: Detection,
        video_processing_start_time: str, # ISO format timestamp
        frame_image_base64: Optional[str] = None # New parameter for the base64 image
    ):
        """
        Logs an individual detection from a video frame to the video_detections collection.
        Includes the base64-encoded image of the frame if provided.
        """
        try:
            detection_entry = {
                "video_id": video_id,
                "camera_id": camera_id,
                "camera_name": camera_name,
                "frame_number": frame_number,
                "detection_timestamp": datetime.fromisoformat(detection.timestamp), # Store as ISODate
                "label": detection.label,
                "confidence": detection.confidence,
                "bbox": detection.bbox, # Store bbox as array directly
                "video_processing_start_time": datetime.fromisoformat(video_processing_start_time) # Store as ISODate
            }
            if frame_image_base64:
                detection_entry["frame_image_base64"] = frame_image_base64

            await self.video_detections_collection.insert_one(detection_entry)
            # logger.debug(f"Logged video detection for video {video_id}, frame {frame_number}: {detection.label}")
        except Exception as e:
            logger.error(f"Error logging video detection to MongoDB: {e}")

    async def get_threat_logs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Retrieves recent threat logs from the threat_logs collection.
        """
        try:
            # Sort by timestamp descending, limit, and convert cursor to list
            logs_cursor = self.threat_logs_collection.find({}).sort("timestamp", -1).limit(limit)
            logs = await logs_cursor.to_list(length=limit)
            # Convert ObjectId to string and datetime to ISO string for JSON serialization
            for log in logs:
                log["_id"] = str(log["_id"])
                if isinstance(log["timestamp"], datetime):
                    log["timestamp"] = log["timestamp"].isoformat()
            return logs
        except Exception as e:
            logger.error(f"Error retrieving threat logs from MongoDB: {e}")
            return []

    async def get_video_detections(self, video_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves all detections for a specific video_id from the video_detections collection.
        """
        try:
            detections_cursor = self.video_detections_collection.find({"video_id": video_id}).sort([("frame_number", 1), ("detection_timestamp", 1)])
            detections = await detections_cursor.to_list(length=None) # Get all results
            # Convert ObjectId to string and datetime to ISO string
            for det in detections:
                det["_id"] = str(det["_id"])
                if isinstance(det["detection_timestamp"], datetime):
                    det["detection_timestamp"] = det["detection_timestamp"].isoformat()
                if isinstance(det["video_processing_start_time"], datetime):
                    det["video_processing_start_time"] = det["video_processing_start_time"].isoformat()
            return detections
        except Exception as e:
            logger.error(f"Error retrieving video detections for video_id {video_id} from MongoDB: {e}")
            return []


class ConnectionManager:
    """Manages WebSocket connections for real-time detection streaming."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {} # Store connections by camera_id
        self.camera_threat_states: Dict[str, Dict[str, Any]] = {} # Track threat state per camera
        # Initialize DB manager here, passing MongoDB URI and DB name from env
        mongo_uri = os.getenv("MONGO_URI")
        mongo_db_name = os.getenv("MONGO_DB_NAME")
        if not mongo_uri or not mongo_db_name:
            # This should ideally be caught during startup, but as a fallback
            logger.critical("MONGO_URI or MONGO_DB_NAME not found in environment variables. Database will not function.")
            raise ValueError("MONGO_URI and MONGO_DB_NAME must be set in .env for MongoDB.")
        self.db_manager = DatabaseManager(mongo_uri, mongo_db_name) 
        logger.info("ConnectionManager initialized")

    async def connect(self, websocket: WebSocket, camera_id: str):
        await websocket.accept()
        self.active_connections[camera_id] = websocket
        # Initialize threat state for new camera
        self.camera_threat_states[camera_id] = {
            "is_threat_active": False,
            "last_threat_timestamp": None,
            "last_log_action": None # To prevent duplicate entry/exit logs
        }
        logger.info(f"Client connected for camera {camera_id}. Total connections: {len(self.active_connections)}")

    def disconnect(self, camera_id: str):
        if camera_id in self.active_connections:
            del self.active_connections[camera_id]
            if camera_id in self.camera_threat_states:
                # Log exit if a threat was active when camera disconnected
                if self.camera_threat_states[camera_id]["is_threat_active"]:
                    # Schedule the async DB operation without blocking the disconnect
                    asyncio.create_task(self.db_manager.log_threat_event(camera_id, "Unknown Camera", "exit")) # Use "Unknown Camera" as name
                del self.camera_threat_states[camera_id]
            logger.info(f"Client disconnected for camera {camera_id}. Total connections: {len(self.active_connections)}")
        
    async def send_detection_result(self, websocket: WebSocket, data: dict):
        try:
            await websocket.send_text(json.dumps(data))
        except Exception as e:
            logger.error(f"Error sending data to client for camera {data.get('camera_id')}: {str(e)}")
            self.disconnect(data.get('camera_id'))

    async def update_threat_state_and_log(self, camera_id: str, camera_name: str, threat_detected_in_frame: bool, confidence: Optional[float] = None):
        """
        Updates the threat state for a camera and logs events to the database.
        This method is now async because it interacts with the async db_manager.
        """
        current_time = datetime.now()
        state = self.camera_threat_states.get(camera_id)
        if not state:
            logger.warning(f"No state found for camera {camera_id}. Cannot log threat event.")
            return

        is_threat_active = state["is_threat_active"]
        last_log_action = state["last_log_action"]

        if threat_detected_in_frame:
            state["last_threat_timestamp"] = current_time
            if not is_threat_active:
                # Threat just started or re-appeared
                state["is_threat_active"] = True
                if last_log_action != "entry":
                    await self.db_manager.log_threat_event(camera_id, camera_name, "entry", confidence) # Await this call
                    state["last_log_action"] = "entry"
        else:
            # Check if threat should still be considered active due to persistence
            if is_threat_active and state["last_threat_timestamp"] and \
               (current_time - state["last_threat_timestamp"]) < timedelta(seconds=5):
                # Threat is still active due to 5-second persistence
                pass
            else:
                # Threat is no longer active (either never was, or 5-second window expired)
                if is_threat_active: # Only log exit if it was previously active
                    if last_log_action != "exit":
                        await self.db_manager.log_threat_event(camera_id, camera_name, "exit") # Await this call
                        state["last_log_action"] = "exit"
                state["is_threat_active"] = False
                state["last_threat_timestamp"] = None # Reset timestamp when no longer active

class LiveWeaponDetectionAPI:
    """
    Live Weapon Detection API using YOLOv8 model for real-time threat detection
    from video streams.
    """
    
    def __init__(self):
        logger.info("Initializing LiveWeaponDetectionAPI...")
        
        self.live_model = None
        self.video_model = None
        
        self.live_model_path = os.getenv("LIVE_MODEL_PATH")
        self.video_model_path = os.getenv("VIDEO_MODEL_PATH")
        self.allowed_origins = os.getenv("ALLOWED_ORIGINS", "").split(",")
        self.confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.49"))
        self.max_detections = int(os.getenv("MAX_DETECTIONS", "10"))
        
        # Performance optimization settings
        self.input_size = int(os.getenv("INPUT_SIZE", "416"))  # Smaller input size for speed
        self.skip_frames = int(os.getenv("SKIP_FRAMES", "1"))  # Process every N frames
        self.frame_counter = 0
        self.last_result = None  # Store last detection result for skipped frames
        
        # Initialize performance tracking
        try:
            self.stats = PerformanceStats()
            logger.info("Performance stats initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize performance stats: {str(e)}")
            # Create a simple fallback stats object
            self.stats = type('obj', (object,), {
                'total_frames': 0,
                'total_detections': 0,
                'total_processing_time': 0.0,
                'start_time': datetime.now(),
                'update': lambda pt, dc: None, # Corrected lambda to not use self
                'get_stats': lambda: {"error": "Stats unavailable"} # Corrected lambda to not use self
            })()
        
        # Validate environment variables for model paths
        if not self.live_model_path:
            raise ValueError("LIVE_MODEL_PATH not found in environment variables")
        if not self.video_model_path:
            raise ValueError("VIDEO_MODEL_PATH not found in environment variables")
        
        # Load both YOLO models
        self._load_models()
        logger.info("LiveWeaponDetectionAPI initialization complete")

    def _load_models(self):
        """Load both the live and video processing YOLO models."""
        self.live_model = self._load_single_model(self.live_model_path, "live")
        self.video_model = self._load_single_model(self.video_model_path, "video")

    def _load_single_model(self, model_path: str, model_type: str):
        """Helper to load a single YOLO model."""
        try:
            # Check if the full model file exists
            if not os.path.exists(model_path):
                logger.warning(f"Model file '{model_path}' for {model_type} not found. Attempting to reassemble from chunks.")
                model_dir = os.path.dirname(model_path)
                model_file_name = os.path.basename(model_path)
                chunk_dir = os.path.join(model_dir, f"{model_file_name}_chunks")

                reassembled_path = reassemble_file(chunk_dir, model_file_name, model_dir)
                
                if not reassembled_path or not os.path.exists(reassembled_path):
                    logger.error(f"Failed to reassemble {model_type} model from chunks in {chunk_dir}. Using dummy model.")
                    return self._get_dummy_model()
            
            model = YOLO(model_path)
            model.fuse()  # Optimize model for inference speed
            
            # Warm up the model with a dummy image
            dummy_img = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
            _ = model(dummy_img, verbose=False, conf=self.confidence_threshold, imgsz=self.input_size)
            
            logger.info(f"{model_type.capitalize()} model loaded and optimized successfully from: {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load {model_type} model from {model_path}: {str(e)}")
            # Return dummy model if real model fails to load
            return self._get_dummy_model()

    def _get_dummy_model(self):
        """Returns a dummy YOLO model for fallback scenarios."""
        class DummyYOLOModel:
            def __call__(self, *args, **kwargs):
                # Simulate a detection if 'weapon' is in names and a condition is met
                if self.names.get(0) == 'weapon' and datetime.now().second % 5 == 0:
                    class DummyBoxes:
                        def __init__(self):
                            self.xyxy = np.array([[100, 100, 200, 200]]) # Dummy bbox
                            self.conf = np.array([0.75]) # Dummy confidence
                            self.cls = np.array([0]) # Dummy class ID
                        def __len__(self):
                            return 1
                    
                    class DummyResult:
                        def __init__(self):
                            self.boxes = DummyBoxes()
                    
                    return [DummyResult()]
                return [type('obj', (object,), {'boxes': None})()] # No detections
            def fuse(self):
                pass
            @property
            def names(self):
                return {0: 'weapon'}
        return DummyYOLOModel()

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for faster inference.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            np.ndarray: Preprocessed image
        """
        # Resize image to model input size for faster processing
        height, width = image.shape[:2]
        
        # Only resize if image is larger than input size
        if max(height, width) > self.input_size:
            # Calculate scaling factor to maintain aspect ratio
            scale = self.input_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return image
    
    def _decode_base64_image(self, base64_string: str) -> np.ndarray:
        """
        Decode base64 image string to OpenCV format.
        
        Args:
            base64_string: Base64 encoded image string
            
        Returns:
            np.ndarray: Decoded image in BGR format
        """
        try:
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # Decode base64 string
            image_bytes = base64.b64decode(base64_string)
            
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            
            # Decode image
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Invalid image format or corrupted image data")
            
            return image
            
        except Exception as e:
            logger.error(f"Base64 image decoding failed: {str(e)}")
            raise ValueError(f"Failed to decode base64 image: {str(e)}")
        
    def _run_inference(self, image: np.ndarray, frame_id: int = None, camera_id: str = "unknown_camera", camera_name: str = "Unknown Camera") -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
        """
        Run YOLO inference on the input image frame with performance optimizations.
        This method is designed to be called from both sync (video processing) and async (websocket) contexts.
        The DB logging part for live streams is scheduled as a separate task to avoid blocking.
        
        Returns:
            Tuple[Dict[str, Any], Optional[np.ndarray]]: A tuple containing the detection results
            dictionary and the processed image (with bounding boxes if applicable and for video processing).
            The image is None for live camera streams (as frontend handles drawing).
        """
        try:
            # Determine which model to use
            current_model = self.video_model if camera_id == "video_processor" else self.live_model

            # This frame_counter and last_result logic is primarily for WebSocket streaming
            # For video processing, we'll process every frame or apply a different skipping logic.
            # Keeping it here for consistency with existing WebSocket logic.
            if frame_id is not None and camera_id != "video_processor": # Only apply skipping for WebSocket frames with a frame_id
                self.frame_counter += 1
                if self.frame_counter % self.skip_frames != 0 and self.last_result is not None:
                    cached_result = self.last_result.copy()
                    cached_result["frame_id"] = frame_id
                    cached_result["timestamp"] = datetime.now().isoformat()
                    cached_result["cached"] = True
                    cached_result["camera_id"] = camera_id
                    cached_result["camera_name"] = camera_name
                    return cached_result, None # Return None for image for cached live frames
            
            start_time = datetime.now()
            
            # Preprocess image for faster inference
            processed_image = self._preprocess_image(image)
            
            # Convert BGR to RGB for YOLO model
            image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            
            detections = []
            total_boxes = 0
            threat_detected_in_frame = False
            max_confidence = 0.0
            
            # Initialize image for drawing if it's a video processing task
            image_with_boxes = processed_image.copy() if camera_id == "video_processor" else None

            # Run inference with optimized settings
            # Check if model is a dummy model
            if isinstance(current_model, type('obj', (object,), {})) and hasattr(current_model, '__call__') and not hasattr(current_model, 'fuse'):
                # This is our dummy model, simulate detections based on its __call__ method
                results = current_model(image_rgb)
            else:
                # Real YOLO model inference
                results = current_model(
                    image_rgb, 
                    verbose=False, 
                    conf=self.confidence_threshold,
                    imgsz=self.input_size,
                    half=True,  # Use FP16 for faster inference if supported
                    device='cpu'  # Explicitly set device
                )
            
            # Process results
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes
                    total_boxes = len(boxes)
                    
                    # Limit processing to avoid overload
                    max_process = min(len(boxes), self.max_detections)
                    
                    for i in range(max_process):
                        # Get confidence score first for early filtering
                        confidence = float(boxes.conf[i].cpu().numpy())
                        
                        # Skip low confidence detections early
                        if confidence < self.confidence_threshold:
                            continue
                        
                        # Get bounding box coordinates [x1, y1, x2, y2]
                        bbox = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                        
                        # Get class label
                        class_id = int(boxes.cls[i].cpu().numpy())
                        label = current_model.names[class_id] # Use current_model's names
                        
                        detection = Detection(
                            label=label,
                            confidence=round(confidence, 2),
                            bbox=bbox,
                            timestamp=datetime.now().isoformat()
                        )
                        
                        detections.append(detection)
                        threat_detected_in_frame = True
                        if confidence > max_confidence:
                            max_confidence = confidence

                        # Draw bounding box and label on the image if it's for video processing
                        if image_with_boxes is not None:
                            x1, y1, x2, y2 = bbox
                            color = (0, 0, 255) # Red color for bounding box (BGR)
                            thickness = 2
                            font_scale = 0.7
                            font_thickness = 2
                            
                            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, thickness)
                            cv2.putText(image_with_boxes, f"{label} {confidence:.2f}", (x1, y1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness, cv2.LINE_AA)
            
            # Calculate processing time and FPS
            processing_time = (datetime.now() - start_time).total_seconds()
            fps = 1.0 / processing_time if processing_time > 0 else 0
            
            # Update performance statistics safely
            try:
                self.stats.update(processing_time, len(detections))
            except Exception as e:
                logger.warning(f"Failed to update stats: {str(e)}")

            # Update threat state and log to DB (only for live camera streams)
            # This is scheduled as a task to avoid blocking the inference loop
            if camera_id != "video_processor":
                asyncio.create_task(connection_manager.update_threat_state_and_log(
                    camera_id, camera_name, threat_detected_in_frame, max_confidence if threat_detected_in_frame else None
                ))
            
            response = {
                "detections": [det.dict() for det in detections],
                "frame_id": frame_id,
                "processing_time": round(processing_time, 4),
                "fps": round(fps, 1),
                "timestamp": datetime.now().isoformat(),
                "threat_detected": threat_detected_in_frame, # This now reflects current frame only
                "total_boxes_detected": total_boxes,
                "boxes_after_filtering": len(detections),
                "cached": False,
                "frame_counter": self.frame_counter,
                "camera_id": camera_id, # Include camera_id in response
                "camera_name": camera_name # Include camera_name in response
            }
            
            # Cache result for frame skipping (only for live camera streams)
            if frame_id is not None and camera_id != "video_processor":
                self.last_result = response.copy() 
            
            if detections:
                log_prefix = f"Frame {frame_id} from {camera_id}" if frame_id is not None else f"Video Frame from {camera_id}"
                logger.info(f"{log_prefix}: Detected {len(detections)} weapons in {processing_time:.4f}s ({fps:.1f} FPS)")
            
            return response, image_with_boxes # Return the image if it's a video frame
            
        except Exception as e:
            logger.error(f"Inference failed for camera {camera_id}: {str(e)}")
            raise RuntimeError(f"Model inference failed for camera {camera_id}: {str(e)}")

    async def process_video_file(self, video_path: str, video_id: str, camera_id: str, camera_name: str) -> VideoProcessResponse:
        """
        Processes a video file frame by frame for weapon detection.
        """
        logger.info(f"Starting video processing for video_id: {video_id}, path: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Error: Could not open video file {video_path}")
            raise HTTPException(status_code=400, detail=f"Could not open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_video = cap.get(cv2.CAP_PROP_FPS)
        
        detections_by_frame: List[VideoFrameDetection] = []
        overall_total_detections = 0
        frame_number = 0
        video_processing_start_time = datetime.now().isoformat()
        processing_start_time = datetime.now()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break # End of video

                frame_number += 1
                
                # Run inference on the frame, getting back the detections and the drawn frame
                detection_result, frame_with_boxes = self._run_inference(frame, frame_id=frame_number, camera_id=camera_id, camera_name=camera_name)
                
                # Collect detections for this frame
                frame_detections = []
                if detection_result.get("detections"):
                    # Only process and store frames if detections were found
                    for det_data in detection_result.get("detections", []):
                        detection = Detection(**det_data)
                        frame_detections.append(detection)
                        overall_total_detections += 1
                        
                        # Log to console for real-time feedback during video processing
                        logger.info(f"Video {video_id}, Frame {frame_number}: Detected {detection.label} with {detection.confidence:.2f} confidence at {detection.bbox}")

                    # Convert frame with boxes to base64 for storage
                    _, buffer = cv2.imencode('.jpg', frame_with_boxes)
                    frame_image_base64 = base64.b64encode(buffer).decode('utf-8')

                    # Log each individual detection to the database, including the image
                    await connection_manager.db_manager.log_video_detection( # Await this call
                        video_id=video_id,
                        camera_id=camera_id,
                        camera_name=camera_name,
                        frame_number=frame_number,
                        detection=detection,
                        video_processing_start_time=video_processing_start_time,
                        frame_image_base64=frame_image_base64 # Pass the base64 image
                    )
                    
                    detections_by_frame.append(VideoFrameDetection(
                        frame_number=frame_number,
                        timestamp=datetime.now().isoformat(), # Timestamp when this frame was processed
                        detections=frame_detections,
                        frame_image_base64=frame_image_base64 # Also include in the response
                    ))

        finally:
            cap.release()
            
        processing_duration = (datetime.now() - processing_start_time).total_seconds()
        logger.info(f"Finished video processing for video_id: {video_id}. Total frames: {frame_number}, Detections: {overall_total_detections}, Duration: {processing_duration:.2f}s")

        return VideoProcessResponse(
            video_id=video_id,
            status="completed",
            total_frames_processed=frame_number, # This is total frames iterated, not just detected
            total_detections=overall_total_detections,
            processing_duration_seconds=round(processing_duration, 2),
            detections_by_frame=detections_by_frame,
            message=f"Video processing completed. Detected {overall_total_detections} threats across {len(detections_by_frame)} frames."
        )


# Initialize the detection API and connection manager
# These need to be initialized synchronously for app.add_middleware
# to have access to detection_api.allowed_origins
# We will then re-check MongoDB connection and create indexes in startup_event
try:
    logger.info("Starting initial synchronous application component setup...")
    detection_api = LiveWeaponDetectionAPI() # This will now load both models
    connection_manager = ConnectionManager() # This will initialize DBManager

    logger.info("Initial synchronous application component setup complete.")
except Exception as e:
    logger.critical(f"FATAL ERROR: Failed to initialize core components synchronously: {str(e)}")
    # Exit if core components cannot be initialized
    import sys
    sys.exit(1)

# Create FastAPI application
app = FastAPI(
    title="Live Weapon Detection API",
    description="Real-time weapon detection from live video streams and batch video files using YOLOv8",
    version="1.0.0"
)

# Configure CORS - now `detection_api.allowed_origins` is available
app.add_middleware(
    CORSMiddleware,
    allow_origins=detection_api.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Add startup event handler to perform async initializations (DB connection, indexes)
@app.on_event("startup")
async def startup_event():
    logger.info("Running FastAPI startup event...")
    # Check MongoDB connection and create indexes
    if not await connection_manager.db_manager._check_connection():
        logger.critical("MongoDB connection failed during startup. Database operations may not work.")
        # Optionally, raise an exception here to prevent server from fully starting
        # raise RuntimeError("Database connection failed.")
    else:
        await connection_manager.db_manager._create_indexes() # Ensure indexes are created
    logger.info("FastAPI startup event complete.")

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "Live Weapon Detection API is running",
        "live_model_loaded": detection_api.live_model is not None,
        "video_model_loaded": detection_api.video_model is not None,
        "live_model_path": detection_api.live_model_path,
        "video_model_path": detection_api.video_model_path,
        "active_connections": len(connection_manager.active_connections),
        "confidence_threshold": detection_api.confidence_threshold
    }

@app.get("/health")
async def health_check():
    """Detailed health check endpoint."""
    return {
        "status": "healthy",
        "live_model_loaded": detection_api.live_model is not None,
        "video_model_loaded": detection_api.video_model is not None,
        "live_model_path": detection_api.live_model_path,
        "video_model_path": detection_api.video_model_path,
        "allowed_origins": detection_api.allowed_origins,
        "active_connections": len(connection_manager.active_connections),
        "confidence_threshold": detection_api.confidence_threshold,
        "max_detections": detection_api.max_detections
    }

@app.get("/stats")
async def get_performance_stats():
    """Get performance statistics."""
    try:
        performance_stats = detection_api.stats.get_stats()
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        performance_stats = {"error": "Stats unavailable"}
    
    return {
        "status": "active",
        "performance": performance_stats,
        "configuration": {
            "confidence_threshold": detection_api.confidence_threshold,
            "max_detections": detection_api.max_detections,
            "live_model_path": detection_api.live_model_path,
            "video_model_path": detection_api.video_model_path
        },
        "connections": {
            "active_connections": len(connection_manager.active_connections)
        }
    }

@app.post("/model/chunk")
async def chunk_models(request: ChunkRequest):
    """
    Endpoint to trigger chunking of multiple model files.
    """
    results = {}
    for model_path in request.model_paths:
        try:
            chunks = chunk_file(model_path, request.chunk_size_mb)
            results[model_path] = {"status": "success", "chunks_created": len(chunks)}
        except FileNotFoundError:
            results[model_path] = {"status": "failed", "error": f"Model file not found at {model_path}"}
        except Exception as e:
            logger.error(f"Error chunking model {model_path}: {e}")
            results[model_path] = {"status": "failed", "error": f"Failed to chunk model: {str(e)}"}
    
    return {"message": "Chunking process completed for specified models.", "results": results}

@app.post("/process_video", response_model=VideoProcessResponse)
async def process_video(
    video_file: Optional[UploadFile] = File(None),
    video_url: Optional[str] = Form(None),
    camera_id: str = Form("video_processor"),
    camera_name: str = Form("Video Processor")
):
    """
    API endpoint to process a whole video for weapon detection.
    Accepts either a video file upload or a video URL.
    """
    if not video_file and not video_url:
        raise HTTPException(status_code=400, detail="Either 'video_file' or 'video_url' must be provided.")

    temp_video_path = None
    video_id = str(uuid.uuid4()) # Generate a unique ID for this video processing task

    try:
        if video_file:
            # Save the uploaded file temporarily
            temp_video_path = f"temp_video_{video_id}_{video_file.filename}"
            with open(temp_video_path, "wb") as buffer:
                while True:
                    chunk = await video_file.read(1024 * 1024) # Read in 1MB chunks
                    if not chunk:
                        break
                    buffer.write(chunk)
            logger.info(f"Received uploaded video: {video_file.filename}, saved to {temp_video_path}")
            
            # Process the temporary video file
            response = await detection_api.process_video_file(temp_video_path, video_id, camera_id, camera_name)
            
        elif video_url:
            # For simplicity, this example assumes video_url is directly readable by OpenCV.
            # In a real-world scenario, you might need to download the video first.
            logger.info(f"Processing video from URL: {video_url}")
            response = await detection_api.process_video_file(video_url, video_id, camera_id, camera_name)
            
        return response

    except HTTPException as e:
        raise e # Re-raise FastAPI HTTPExceptions
    except Exception as e:
        logger.error(f"Error processing video {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process video: {str(e)}")
    finally:
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            logger.info(f"Cleaned up temporary video file: {temp_video_path}")

@app.get("/logs/threats")
async def get_threat_logs(limit: int = 20):
    """
    Endpoint to retrieve recent threat logs from the database.
    """
    try:
        logs = await connection_manager.db_manager.get_threat_logs(limit=limit) # Await this call
        return {"logs": logs}
    except Exception as e:
        logger.error(f"Error retrieving threat logs from database: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve logs: {e}")

@app.get("/logs/video_detections/{video_id}", response_model=List[VideoFrameDetection])
async def get_video_detections_by_id(video_id: str):
    """
    Endpoint to retrieve detailed detections for a specific video ID, including base64 images of detected frames.
    """
    try:
        detections_raw = await connection_manager.db_manager.get_video_detections(video_id) # Await this call
        if not detections_raw:
            raise HTTPException(status_code=404, detail=f"No detections found for video ID: {video_id}")
        
        # Group detections by frame number for a cleaner response
        grouped_detections: Dict[int, List[Detection]] = {}
        frame_images: Dict[int, str] = {} # Store the base64 image for each frame number
        
        for det_row in detections_raw:
            frame_num = det_row["frame_number"]
            if frame_num not in grouped_detections:
                grouped_detections[frame_num] = []
            
            # Reconstruct Detection object from database row
            grouped_detections[frame_num].append(Detection(
                label=det_row["label"],
                confidence=det_row["confidence"],
                bbox=det_row["bbox"], # bbox is already a list from MongoDB
                timestamp=det_row["detection_timestamp"]
            ))
            # Store the frame image if available
            if "frame_image_base64" in det_row and det_row["frame_image_base64"]:
                frame_images[frame_num] = det_row["frame_image_base64"]
        
        video_frame_detections = []
        for fn in sorted(grouped_detections.keys()):
            video_frame_detections.append(VideoFrameDetection(
                frame_number=fn,
                timestamp=grouped_detections[fn][0].timestamp, # Use timestamp of first detection in frame
                detections=grouped_detections[fn],
                frame_image_base64=frame_images.get(fn) # Attach the base64 image
            ))

        return video_frame_detections
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error retrieving video detections for video ID {video_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve video detections: {e}")


@app.get("/demo", response_class=HTMLResponse)
async def demo_page():
    """Simple demo page for testing the live detection and video processing."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Live Weapon Detection Demo</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css" rel="stylesheet">
        <style>
            body { font-family: 'Inter', sans-serif; }
            .camera-feed-container {
                position: relative;
                border: 2px solid #ccc;
                border-radius: 8px;
                overflow: hidden;
                margin: 10px;
                flex-grow: 1;
                min-width: 300px; /* Minimum width for responsiveness */
                max-width: 100%;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .camera-feed-container.threat-alert {
                border-color: #ef4444; /* Tailwind red-500 */
                box-shadow: 0 0 15px rgba(239, 68, 68, 0.7); /* Stronger red shadow */
            }
            .camera-feed-container video, .camera-feed-container canvas {
                width: 100%;
                height: auto;
                display: block;
            }
            .detection-info {
                padding: 10px;
                background: #f9fafb; /* Tailwind gray-50 */
                border-top: 1px solid #e5e7eb; /* Tailwind gray-200 */
            }
            .threat-count {
                font-weight: bold;
                color: #ef4444; /* Tailwind red-500 */
            }
            .popup-container {
                position: fixed;
                bottom: 20px;
                right: 20px;
                z-index: 1000;
                display: flex;
                flex-direction: column-reverse; /* Newest popups appear on top */
                gap: 10px;
            }
            .popup {
                background-color: #fef2f2; /* Tailwind red-50 */
                border: 1px solid #ef4444; /* Tailwind red-500 */
                color: #b91c1c; /* Tailwind red-700 */
                padding: 12px 20px;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                opacity: 0;
                transition: opacity 0.5s ease-in-out, transform 0.5s ease-in-out;
                transform: translateX(100%);
                font-size: 0.9rem;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            .popup.show {
                opacity: 1;
                transform: translateX(0);
            }
            .popup .icon {
                font-size: 1.2rem;
            }

            /* Activity Log Styles */
            .activity-log-container {
                background-color: #1a202c; /* Dark background for the log */
                border-radius: 8px;
                padding: 1rem;
                color: #e2e8f0; /* Light text color */
                max-height: 400px; /* Limit height */
                overflow-y: auto; /* Scrollable */
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
            }
            .log-entry {
                display: flex;
                align-items: center;
                padding: 0.75rem 0;
                border-bottom: 1px solid #2d3748; /* Darker border */
            }
            .log-entry:last-child {
                border-bottom: none;
            }
            .log-icon {
                font-size: 1.2rem;
                margin-right: 0.75rem;
                width: 24px; /* Fixed width for alignment */
                text-align: center;
            }
            .log-icon.entry {
                color: #48bb78; /* Green for entry */
            }
            .log-icon.exit {
                color: #f6ad55; /* Orange for exit */
            }
            .log-content {
                flex-grow: 1;
            }
            .log-camera-name {
                font-weight: bold;
                color: #cbd5e0; /* Lighter text for name */
            }
            .log-action {
                font-size: 0.85rem;
                padding: 0.1rem 0.4rem;
                border-radius: 4px;
                font-weight: bold;
                margin-left: 0.5rem;
            }
            .log-action.entry {
                background-color: #2f855a; /* Darker green */
                color: #fff;
            }
            .log-action.exit {
                background-color: #dd6b20; /* Darker orange */
                color: #fff;
            }
            .log-confidence {
                font-size: 0.8rem;
                color: #a0aec0; /* Gray text */
            }
            .log-timestamp {
                font-size: 0.8rem;
                color: #a0aec0; /* Gray text */
                margin-left: auto; /* Push to right */
            }
            /* Video Processing Section */
            .video-upload-container {
                background-color: #fff;
                border-radius: 8px;
                padding: 1.5rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .video-results-container {
                background-color: #f0f4f8; /* Light blue-gray */
                border-radius: 8px;
                padding: 1.5rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                max-height: 500px;
                overflow-y: auto;
            }
            .detected-frame-card {
                background-color: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 1rem;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
                margin-bottom: 1rem;
            }
            .detected-frame-card img {
                max-width: 100%;
                height: auto;
                border-radius: 4px;
                margin-top: 0.5rem;
                border: 1px solid #cbd5e0;
            }
        </style>
    </head>
    <body class="bg-gray-100 text-gray-900 p-4 sm:p-6 lg:p-8">
        <h1 class="text-3xl font-bold text-center mb-6 text-gray-800">Live Weapon Detection Demo</h1>
        
        <div class="flex flex-col sm:flex-row items-center justify-center mb-6 space-y-4 sm:space-y-0 sm:space-x-4">
            <div class="relative inline-block text-left w-full sm:w-auto">
                <select id="cameraSelect" class="block appearance-none w-full bg-white border border-gray-300 hover:border-gray-400 px-4 py-2 pr-8 rounded-md shadow leading-tight focus:outline-none focus:shadow-outline text-gray-700">
                    <option value="">Select Camera</option>
                </select>
                <div class="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-700">
                    <svg class="fill-current h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z"/></svg>
                </div>
            </div>
            <button onclick="addCamera()" class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-6 rounded-md shadow-lg transition duration-300 ease-in-out transform hover:scale-105 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-75">
                <i class="fas fa-plus-circle mr-2"></i> Add Camera
            </button>
        </div>

        <div class="flex flex-col lg:flex-row gap-6">
            <div id="camera-container" class="flex flex-wrap justify-center items-start gap-4 p-2 lg:w-2/3">
                <!-- Camera feeds will be added here dynamically -->
            </div>

            <div class="lg:w-1/3 p-4">
                <div class="activity-log-container mb-6">
                    <h2 class="text-xl font-bold mb-4 text-gray-200 flex items-center">
                        <i class="fas fa-history mr-2"></i> Recent Activity
                        <span id="refreshLogs" class="ml-auto text-sm text-blue-400 cursor-pointer hover:underline">
                            <i class="fas fa-sync-alt"></i> 4 mins ago
                        </span>
                    </h2>
                    <div id="activityLogList">
                        <!-- Activity logs will be loaded here -->
                        <p class="text-center text-gray-500">Loading activity logs...</p>
                    </div>
                </div>

                <div class="video-upload-container">
                    <h2 class="text-xl font-bold mb-4 text-gray-800 flex items-center">
                        <i class="fas fa-video mr-2"></i> Process Video File
                    </h2>
                    <input type="file" id="videoFileInput" accept="video/*" class="block w-full text-sm text-gray-900 border border-gray-300 rounded-lg cursor-pointer bg-gray-50 focus:outline-none mb-4 p-2">
                    <button id="processVideoButton" class="bg-purple-600 hover:bg-purple-700 text-white font-bold py-2 px-6 rounded-md shadow-lg transition duration-300 ease-in-out transform hover:scale-105 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-opacity-75 w-full">
                        <i class="fas fa-cogs mr-2"></i> Process Video
                    </button>
                    <div id="videoProcessingStatus" class="mt-4 text-center text-sm text-gray-600"></div>
                    
                    <div id="videoResults" class="video-results-container mt-6 hidden">
                        <h3 class="text-lg font-bold mb-3 text-gray-800">Video Processing Results: <span id="videoResultId" class="font-normal text-gray-600"></span></h3>
                        <p class="text-sm text-gray-700 mb-2">Total Frames: <span id="videoTotalFrames"></span></p>
                        <p class="text-sm text-gray-700 mb-2">Total Detections: <span id="videoTotalDetections"></span></p>
                        <p class="text-sm text-gray-700 mb-4">Duration: <span id="videoDuration"></span>s</p>
                        <button id="viewDetectedFramesButton" class="bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded-md shadow-lg transition duration-300 ease-in-out focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-opacity-75 w-full mt-4">
                            <i class="fas fa-eye mr-2"></i> View Detected Frames
                        </button>
                        <div id="videoDetectionsList" class="space-y-3 mt-4">
                            <!-- Video detections will be listed here -->
                        </div>
                    </div>
                </div>
            </div>
        </div>


        <div id="popup-container" class="popup-container">
            <!-- Popups will appear here -->
        </div>

        <script>
            // Load backend URL from environment (simulated for browser environment)
            // In a real build process (e.g., React with .env), this would be handled automatically.
            // For this HTML demo, we'll assume it's directly available or set via a build step.
            // For local testing, it defaults to http://localhost:8005
            const BACKEND_URL = window.location.origin; // Dynamically get current origin
            // You can override this for testing if needed:
            // const BACKEND_URL = "http://localhost:8005"; 

            let cameraCounter = 0;
            const activeCameras = {}; // Stores CameraInstance objects keyed by camera_id
            const THREAT_PERSISTENCE_MS = 5000; // 5 seconds
            const POPUP_COOLDOWN_MS = 5000; // 5 seconds cooldown for popups
            const LOG_REFRESH_INTERVAL_MS = 60000; // Refresh logs every 60 seconds

            class CameraInstance {
                constructor(deviceId, deviceName) {
                    this.id = `camera-${++cameraCounter}`;
                    this.deviceId = deviceId;
                    this.name = deviceName || `Camera ${cameraCounter}`;
                    this.video = null;
                    this.canvas = null;
                    this.ctx = null;
                    this.ws = null;
                    this.isDetecting = false;
                    this.threatDetectedUntil = 0; // Timestamp when threat alert should expire
                    this.currentThreatCount = 0;
                    this.lastFrameSentTime = 0;
                    this.frameSendInterval = 200; // Send frame every 200ms (5 FPS)
                    this.lastPopupTime = 0; // New: Track last popup time for this camera

                    this.createElement();
                    this.initCameraStream();
                }

                createElement() {
                    const container = document.getElementById('camera-container');
                    this.element = document.createElement('div');
                    this.element.id = `camera-card-${this.id}`;
                    this.element.className = 'camera-feed-container flex flex-col';
                    this.element.innerHTML = `
                        <h3 class="text-lg font-semibold p-3 bg-gray-50 border-b border-gray-200">${this.name}</h3>
                        <video id="video-${this.id}" autoplay muted class="w-full h-auto rounded-t-lg"></video>
                        <canvas id="canvas-${this.id}" class="hidden"></canvas>
                        <div class="detection-info p-3">
                            <p class="text-sm text-gray-600">Status: <span id="status-${this.id}" class="font-medium text-gray-700">Initializing...</span></p>
                            <p class="text-sm text-gray-600">Threats: <span id="threatCount-${this.id}" class="threat-count">0</span></p>
                            <p class="text-xs text-gray-500">Proc. Time: <span id="processingTime-${this.id}">0ms</span> | FPS: <span id="fps-${this.id}">0</span></p>
                            <div id="detectionsList-${this.id}" class="text-xs text-gray-500 mt-2"></div>
                            <button onclick="activeCameras['${this.id}'].stopDetection()" class="mt-3 bg-red-500 hover:bg-red-600 text-white text-sm font-bold py-1 px-3 rounded-md transition duration-300 ease-in-out focus:outline-none focus:ring-2 focus:ring-red-400">
                                Stop
                            </button>
                        </div>
                    `;
                    container.appendChild(this.element);

                    this.video = this.element.querySelector(`#video-${this.id}`);
                    this.canvas = this.element.querySelector(`#canvas-${this.id}`);
                    this.ctx = this.canvas.getContext('2d');
                }

                async initCameraStream() {
                    try {
                        const stream = await navigator.mediaDevices.getUserMedia({
                            video: { deviceId: this.deviceId ? { exact: this.deviceId } : undefined }
                        });
                        this.video.srcObject = stream;
                        this.video.onloadedmetadata = () => {
                            this.canvas.width = this.video.videoWidth;
                            this.canvas.height = this.video.videoHeight;
                            this.startDetection();
                        };
                        this.updateStatus('Camera Ready', 'text-green-600');
                    } catch (err) {
                        console.error(`Error accessing camera ${this.name} (${this.deviceId}):`, err);
                        this.updateStatus('Error Accessing Camera', 'text-red-600');
                        showPopup(`Error: Could not access ${this.name}.`, 'error');
                        this.element.remove(); // Remove the card if camera access fails
                        delete activeCameras[this.id];
                    }
                }

                startDetection() {
                    if (this.isDetecting) return;

                    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    const wsUrl = `${wsProtocol}//${window.location.host}/ws/detect?camera_id=${this.id}`; // Pass camera_id

                    this.ws = new WebSocket(wsUrl);

                    this.ws.onopen = () => {
                        this.updateStatus('Connected', 'text-blue-600');
                        this.isDetecting = true;
                        this.sendFrames();
                    };

                    this.ws.onmessage = (event) => {
                        const data = JSON.parse(event.data);
                        this.updateDetectionDisplay(data);
                    };

                    this.ws.onclose = () => {
                        this.updateStatus('Disconnected', 'text-gray-500');
                        this.isDetecting = false;
                        this.currentThreatCount = 0; // Reset threat count on disconnect
                        this.updateThreatUI(); // Update UI to reflect no threats
                        console.log(`Camera ${this.name} disconnected.`);
                    };

                    this.ws.onerror = (error) => {
                        console.error(`WebSocket error for camera ${this.name}:`, error);
                        this.updateStatus('Error', 'text-red-600');
                        showPopup(`WebSocket Error for ${this.name}.`, 'error');
                        this.isDetecting = false;
                    };
                }

                stopDetection() {
                    if (this.ws) {
                        this.ws.close();
                        this.isDetecting = false;
                        if (this.video && this.video.srcObject) {
                            this.video.srcObject.getTracks().forEach(track => track.stop());
                        }
                        this.element.remove(); // Remove the camera card from UI
                        delete activeCameras[this.id];
                        showPopup(`${this.name} stopped.`, 'info');
                    }
                }

                sendFrames() {
                    if (!this.isDetecting || !this.ws || this.ws.readyState !== WebSocket.OPEN) return;

                    const now = Date.now();
                    if (now - this.lastFrameSentTime < this.frameSendInterval) {
                        requestAnimationFrame(() => this.sendFrames());
                        return;
                    }
                    this.lastFrameSentTime = now;

                    this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
                    const imageData = this.canvas.toDataURL('image/jpeg', 0.7); // Reduced quality for faster transfer

                    this.ws.send(JSON.stringify({
                        type: 'frame',
                        data: imageData,
                        frame_id: now,
                        camera_id: this.id, // Ensure camera_id is sent
                        camera_name: this.name // Send camera name for logging
                    }));

                    requestAnimationFrame(() => this.sendFrames());
                }

                updateDetectionDisplay(data) {
                    const threatCountElement = this.element.querySelector(`#threatCount-${this.id}`);
                    const processingTimeElement = this.element.querySelector(`#processingTime-${this.id}`);
                    const fpsElement = this.element.querySelector(`#fps-${this.id}`);
                    const detectionsListElement = this.element.querySelector(`#detectionsList-${this.id}`);

                    this.currentThreatCount = data.detections.length;
                    threatCountElement.textContent = this.currentThreatCount;
                    processingTimeElement.textContent = (data.processing_time * 1000).toFixed(1) + 'ms';
                    fpsElement.textContent = data.fps + ' FPS';

                    // Clear previous bounding boxes
                    const existingBboxes = this.element.querySelectorAll('.detection-bbox');
                    existingBboxes.forEach(bbox => bbox.remove());

                    if (this.currentThreatCount > 0) {
                        this.threatDetectedUntil = Date.now() + THREAT_PERSISTENCE_MS;
                        this.updateStatus('WEAPON DETECTED!', 'text-red-600');
                        // Rate-limit popups
                        if (Date.now() - this.lastPopupTime > POPUP_COOLDOWN_MS) {
                            showPopup(`Threat detected on ${this.name}!`, 'warning');
                            this.lastPopupTime = Date.now();
                            // Trigger log refresh when a new threat popup is shown
                            fetchThreatLogs(); 
                        }

                        // Draw bounding boxes on the video element
                        data.detections.forEach(detection => {
                            const [x1, y1, x2, y2] = detection.bbox;
                            const videoRect = this.video.getBoundingClientRect();
                            const scaleX = videoRect.width / this.video.videoWidth;
                            const scaleY = videoRect.height / this.video.videoHeight;

                            const bboxElement = document.createElement('div');
                            bboxElement.className = 'detection-bbox';
                            bboxElement.style.left = `${x1 * scaleX}px`;
                            bboxElement.style.top = `${y1 * scaleY}px`;
                            bboxElement.style.width = `${(x2 - x1) * scaleX}px`;
                            bboxElement.style.height = `${(y2 - y1) * scaleY}px`;
                            bboxElement.textContent = `${detection.label} ${(detection.confidence * 100).toFixed(1)}%`;
                            this.element.appendChild(bboxElement);
                        });

                    } else if (Date.now() > this.threatDetectedUntil) {
                        this.updateStatus('Safe', 'text-green-600');
                    }
                    
                    this.updateThreatUI();

                    detectionsListElement.innerHTML = '';
                    data.detections.forEach(detection => {
                        const div = document.createElement('div');
                        div.innerHTML = `<strong>${detection.label}</strong> (${(detection.confidence * 100).toFixed(1)}%)`;
                        detectionsListElement.appendChild(div);
                    });
                }

                updateThreatUI() {
                    if (this.currentThreatCount > 0 || Date.now() < this.threatDetectedUntil) {
                        this.element.classList.add('threat-alert');
                        this.element.querySelector(`#threatCount-${this.id}`).classList.add('text-red-500');
                    } else {
                        this.element.classList.remove('threat-alert');
                        this.element.querySelector(`#threatCount-${this.id}`).classList.remove('text-red-500');
                    }
                }

                updateStatus(message, colorClass) {
                    const statusElement = this.element.querySelector(`#status-${this.id}`);
                    statusElement.textContent = message;
                    statusElement.className = `font-medium ${colorClass}`;
                }
            }

            // Global functions for camera management
            async function initCameraSelection() {
                const cameraSelect = document.getElementById('cameraSelect');
                cameraSelect.innerHTML = '<option value="">Select Camera</option>'; // Reset options

                try {
                    const devices = await navigator.mediaDevices.enumerateDevices();
                    const videoDevices = devices.filter(device => device.kind === 'videoinput');

                    if (videoDevices.length === 0) {
                        showPopup('No video input devices found.', 'error');
                        return;
                    }

                    videoDevices.forEach(device => {
                        const option = document.createElement('option');
                        option.value = device.deviceId;
                        option.textContent = device.label || `Camera ${device.deviceId.substring(0, 8)}`;
                        cameraSelect.appendChild(option);
                    });
                } catch (err) {
                    console.error('Error enumerating devices:', err);
                    showPopup('Error listing cameras. Please grant camera permissions.', 'error');
                }
            }

            function addCamera() {
                const cameraSelect = document.getElementById('cameraSelect');
                const selectedDeviceId = cameraSelect.value;
                const selectedDeviceLabel = cameraSelect.options[cameraSelect.selectedIndex].text;

                if (!selectedDeviceId) {
                    showPopup('Please select a camera from the dropdown.', 'info');
                    return;
                }

                // Check if this camera is already active
                for (const camId in activeCameras) {
                    if (activeCameras[camId].deviceId === selectedDeviceId) {
                        showPopup(`${selectedDeviceLabel} is already active.`, 'info');
                        return;
                    }
                }

                const newCamera = new CameraInstance(selectedDeviceId, selectedDeviceLabel);
                activeCameras[newCamera.id] = newCamera;
                showPopup(`${newCamera.name} added.`, 'success');
            }

            function showPopup(message, type = 'info') {
                const popupContainer = document.getElementById('popup-container');
                const popup = document.createElement('div');
                popup.className = `popup ${type === 'warning' ? 'bg-red-50 border-red-500 text-red-700' : type === 'error' ? 'bg-red-100 border-red-700 text-red-800' : 'bg-blue-50 border-blue-500 text-blue-700'} rounded-md shadow-lg py-2 px-4 flex items-center gap-2`;
                
                let icon = '';
                if (type === 'warning') icon = '<i class="fas fa-exclamation-triangle icon"></i>';
                else if (type === 'error') icon = '<i class="fas fa-times-circle icon"></i>';
                else icon = '<i class="fas fa-info-circle icon"></i>';

                popup.innerHTML = `${icon}<span>${message}</span>`;
                popupContainer.appendChild(popup);

                // Trigger reflow to enable transition
                void popup.offsetWidth; 
                popup.classList.add('show');

                setTimeout(() => {
                    popup.classList.remove('show');
                    popup.addEventListener('transitionend', () => popup.remove());
                }, 5000); // Popup disappears after 5 seconds
            }

            // --- Activity Log Functions ---
            let lastLogRefreshTime = Date.now();

            function formatTimestampForLog(isoTimestamp) {
                const date = new Date(isoTimestamp);
                // Format as HH:MM:SS AM/PM
                return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: true });
            }

            function updateLogRefreshTimeDisplay() {
                const refreshElement = document.getElementById('refreshLogs');
                const now = Date.now();
                const diffMinutes = Math.floor((now - lastLogRefreshTime) / 60000);
                if (diffMinutes === 0) {
                    refreshElement.innerHTML = `<i class="fas fa-sync-alt"></i> just now`;
                } else if (diffMinutes === 1) {
                    refreshElement.innerHTML = `<i class="fas fa-sync-alt"></i> 1 min ago`;
                } else {
                    refreshElement.innerHTML = `<i class="fas fa-sync-alt"></i> ${diffMinutes} mins ago`;
                }
            }

            async function fetchThreatLogs() {
                const activityLogList = document.getElementById('activityLogList');
                activityLogList.innerHTML = '<p class="text-center text-gray-500">Loading activity logs...</p>'; // Show loading

                try {
                    const response = await fetch(`${BACKEND_URL}/logs/threats`);
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    const data = await response.json();
                    const logs = data.logs;

                    activityLogList.innerHTML = ''; // Clear loading message

                    if (logs.length === 0) {
                        activityLogList.innerHTML = '<p class="text-center text-gray-500">No recent activity.</p>';
                        return;
                    }

                    logs.forEach(log => {
                        const logEntry = document.createElement('div');
                        logEntry.className = 'log-entry';
                        
                        const iconClass = log.action === 'entry' ? 'fas fa-arrow-circle-right entry' : 'fas fa-arrow-circle-left exit';
                        const actionClass = log.action === 'entry' ? 'entry' : 'exit';
                        const confidenceDisplay = log.confidence ? ` (${(log.confidence * 100).toFixed(1)}% confidence)` : '';
                        const formattedTime = formatTimestampForLog(log.timestamp);

                        logEntry.innerHTML = `
                            <div class="log-icon"><i class="${iconClass}"></i></div>
                            <div class="log-content">
                                <span class="log-camera-name">${log.camera_name}</span>
                                <span class="log-action ${actionClass}">${log.action.charAt(0).toUpperCase() + log.action.slice(1)}</span>
                                <span class="log-confidence">${confidenceDisplay}</span>
                            </div>
                            <div class="log-timestamp">${formattedTime}</div>
                        `;
                        activityLogList.appendChild(logEntry);
                    });
                    lastLogRefreshTime = Date.now();
                    updateLogRefreshTimeDisplay();

                } catch (error) {
                    console.error('Error fetching threat logs:', error);
                    activityLogList.innerHTML = '<p class="text-center text-red-500">Failed to load activity logs.</p>';
                    showPopup('Failed to load activity logs.', 'error');
                }
            }

            // --- Video Processing Functions ---
            let currentVideoId = null; // To store the video_id for fetching detected frames

            async function processVideoFile() {
                const videoFileInput = document.getElementById('videoFileInput');
                const processVideoButton = document.getElementById('processVideoButton');
                const videoProcessingStatus = document.getElementById('videoProcessingStatus');
                const videoResults = document.getElementById('videoResults');
                const videoDetectionsList = document.getElementById('videoDetectionsList');
                const viewDetectedFramesButton = document.getElementById('viewDetectedFramesButton');


                const file = videoFileInput.files[0];
                if (!file) {
                    showPopup('Please select a video file to process.', 'info');
                    return;
                }

                const formData = new FormData();
                formData.append('video_file', file);
                formData.append('camera_id', 'uploaded_video'); // Fixed ID for uploaded videos
                formData.append('camera_name', file.name); // Use filename as camera name

                processVideoButton.disabled = true;
                processVideoButton.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i> Processing...';
                videoProcessingStatus.textContent = 'Uploading and processing video... This may take a while.';
                videoResults.classList.add('hidden'); // Hide previous results
                videoDetectionsList.innerHTML = ''; // Clear previous detected frames

                try {
                    const response = await fetch(`${BACKEND_URL}/process_video`, {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) {
                        const errorData = await response.json();
                        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
                    }

                    const result = await response.json();
                    videoProcessingStatus.textContent = `Processing complete! ${result.message}`;
                    showPopup('Video processing finished successfully!', 'success');

                    currentVideoId = result.video_id; // Store the video ID

                    // Display summary results
                    document.getElementById('videoResultId').textContent = result.video_id;
                    document.getElementById('videoTotalFrames').textContent = result.total_frames_processed;
                    document.getElementById('videoTotalDetections').textContent = result.total_detections;
                    document.getElementById('videoDuration').textContent = result.processing_duration_seconds.toFixed(2);
                    
                    videoResults.classList.remove('hidden');
                    viewDetectedFramesButton.classList.remove('hidden'); // Show the button
                    fetchThreatLogs(); // Refresh activity logs after video processing

                } catch (error) {
                    console.error('Error processing video:', error);
                    videoProcessingStatus.textContent = `Error: ${error.message}`;
                    showPopup(`Video processing failed: ${error.message}`, 'error');
                } finally {
                    processVideoButton.disabled = false;
                    processVideoButton.innerHTML = '<i class="fas fa-cogs mr-2"></i> Process Video';
                }
            }

            async function viewDetectedFrames() {
                if (!currentVideoId) {
                    showPopup('No video has been processed yet.', 'info');
                    return;
                }

                const videoDetectionsList = document.getElementById('videoDetectionsList');
                videoDetectionsList.innerHTML = '<p class="text-center text-gray-500">Loading detected frames...</p>';

                try {
                    const response = await fetch(`${BACKEND_URL}/logs/video_detections/${currentVideoId}`);
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    const detectedFrames = await response.json();

                    videoDetectionsList.innerHTML = ''; // Clear loading message

                    if (detectedFrames.length === 0) {
                        videoDetectionsList.innerHTML = '<p class="text-center text-gray-500">No threats detected in this video, or no frames stored.</p>';
                        return;
                    }

                    detectedFrames.forEach(frameData => {
                        const frameDiv = document.createElement('div');
                        frameDiv.className = 'detected-frame-card';
                        
                        let detectionsHtml = '';
                        frameData.detections.forEach(det => {
                            detectionsHtml += `<li>${det.label} (${(det.confidence * 100).toFixed(1)}%) at [${det.bbox.join(', ')}]</li>`;
                        });

                        frameDiv.innerHTML = `
                            <p class="font-semibold text-gray-800">Frame ${frameData.frame_number} <span class="text-xs text-gray-500 ml-2">${formatTimestampForLog(frameData.timestamp)}</span></p>
                            <ul class="list-disc list-inside ml-4 text-sm text-gray-700">
                                ${detectionsHtml}
                            </ul>
                            ${frameData.frame_image_base64 ? `<img src="data:image/jpeg;base64,${frameData.frame_image_base64}" alt="Detected Frame ${frameData.frame_number}">` : '<p class="text-sm text-gray-500 mt-2">No image available for this frame.</p>'}
                        `;
                        videoDetectionsList.appendChild(frameDiv);
                    });

                } catch (error) {
                    console.error('Error fetching detected frames:', error);
                    videoDetectionsList.innerHTML = '<p class="text-center text-red-500">Failed to load detected frames.</p>';
                    showPopup(`Failed to load detected frames: ${error.message}`, 'error');
                }
            }


            // Initialize camera selection and fetch logs on page load
            window.onload = () => {
                initCameraSelection();
                fetchThreatLogs();
                setInterval(fetchThreatLogs, LOG_REFRESH_INTERVAL_MS); // Auto-refresh logs
                setInterval(updateLogRefreshTimeDisplay, 10000); // Update "X mins ago" every 10 seconds
                document.getElementById('refreshLogs').addEventListener('click', fetchThreatLogs); // Manual refresh
                document.getElementById('processVideoButton').addEventListener('click', processVideoFile);
                document.getElementById('viewDetectedFramesButton').addEventListener('click', viewDetectedFrames); // New button event listener
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.websocket("/ws/detect")
async def websocket_detection_endpoint(websocket: WebSocket, camera_id: str):
    """
    WebSocket endpoint for real-time weapon detection from live video stream.
    
    Expected message format:
    {
        "type": "frame",
        "data": "base64_encoded_image",
        "frame_id": 12345,
        "camera_id": "unique_camera_id",
        "camera_name": "User-friendly Camera Name"
    }
    """
    await connection_manager.connect(websocket, camera_id)
    
    try:
        while True:
            # Receive frame data from client
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                
                if message.get("type") != "frame":
                    await websocket.send_text(json.dumps({
                        "error": "Invalid message type. Expected 'frame'."
                    }))
                    continue
                
                # Extract frame data
                base64_image = message.get("data")
                frame_id = message.get("frame_id")
                # camera_id is already available from path parameter
                camera_name = message.get("camera_name", "Unknown Camera") # Get camera name from message

                if not base64_image:
                    await websocket.send_text(json.dumps({
                        "error": "No image data provided"
                    }))
                    continue
                
                # Decode and process the image
                try:
                    image = detection_api._decode_base64_image(base64_image)
                    # _run_inference returns (detection_results, processed_image_for_video_only)
                    result, _ = detection_api._run_inference(image, frame_id, camera_id, camera_name)
                    
                    # Send detection results back to client
                    await connection_manager.send_detection_result(websocket, result)
                    
                except ValueError as e:
                    await websocket.send_text(json.dumps({
                        "error": f"Image processing error for camera {camera_id}: {str(e)}",
                        "frame_id": frame_id,
                        "camera_id": camera_id
                    }))
                    
                except RuntimeError as e:
                    await websocket.send_text(json.dumps({
                        "error": f"Detection error for camera {camera_id}: {str(e)}",
                        "frame_id": frame_id,
                        "camera_id": camera_id
                    }))
                
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "error": "Invalid JSON format"
                }))
                
    except WebSocketDisconnect:
        connection_manager.disconnect(camera_id)
    except Exception as e:
        logger.error(f"Unexpected error in WebSocket connection for camera {camera_id}: {str(e)}")
        connection_manager.disconnect(camera_id)

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred"}
    )

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8005))
    
    logger.info(f"Starting Live Weapon Detection server on {host}:{port}")
    logger.info(f"Demo page available at: http://{host}:{port}/demo")
    
    # Uvicorn will automatically call the @app.on_event("startup") function
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
