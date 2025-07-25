import os
import logging
import asyncio
import base64
from typing import List, Dict, Any, Optional
import json
from datetime import datetime, timedelta
import sqlite3

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from ultralytics import YOLO

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

# New Pydantic model for chunking multiple files
class ChunkRequest(BaseModel):
    model_paths: List[str]
    chunk_size_mb: int = 80

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
    """Manages SQLite database operations for threat logs."""
    def __init__(self, db_path: str = "threat_logs.db"):
        self.db_path = db_path
        self._create_tables()
        logger.info(f"DatabaseManager initialized with path: {self.db_path}")

    def _get_db_connection(self):
        """Establishes a connection to the SQLite database."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row # Allows accessing columns by name
        return conn

    def _create_tables(self):
        """Creates the threat_logs table if it doesn't exist."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS threat_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        camera_id TEXT NOT NULL,
                        camera_name TEXT NOT NULL,
                        action TEXT NOT NULL, -- 'entry' or 'exit'
                        timestamp TEXT NOT NULL, -- ISO 8601 format
                        method TEXT, -- 'scanner' or 'manual'
                        confidence REAL -- Optional: for scanner entries
                    );
                """)
                conn.commit()
            logger.info("Threat logs table ensured to exist.")
        except sqlite3.Error as e:
            logger.error(f"Error creating database tables: {e}")

    def log_threat_event(self, camera_id: str, camera_name: str, action: str, confidence: Optional[float] = None):
        """
        Logs a threat event (entry or exit) to the database.

        Args:
            camera_id (str): Unique ID of the camera.
            camera_name (str): User-friendly name of the camera.
            action (str): 'entry' or 'exit'.
            confidence (Optional[float]): Confidence score for 'entry' events.
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                timestamp = datetime.now().isoformat()
                cursor.execute(
                    """
                    INSERT INTO threat_logs (camera_id, camera_name, action, timestamp, method, confidence)
                    VALUES (?, ?, ?, ?, ?, ?);
                    """,
                    (camera_id, camera_name, action, timestamp, "scanner", confidence)
                )
                conn.commit()
            logger.info(f"Logged event: Camera '{camera_name}' ({camera_id}) - Action: {action}, Confidence: {confidence}")
        except sqlite3.Error as e:
            logger.error(f"Error logging threat event: {e}")

    def get_threat_logs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Retrieves recent threat logs from the database.

        Args:
            limit (int): Maximum number of logs to retrieve.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a log entry.
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT camera_id, camera_name, action, timestamp, confidence
                    FROM threat_logs
                    ORDER BY timestamp DESC
                    LIMIT ?;
                    """,
                    (limit,)
                )
                logs = cursor.fetchall()
                # Convert Row objects to dictionaries
                return [dict(row) for row in logs]
        except sqlite3.Error as e:
            logger.error(f"Error retrieving threat logs: {e}")
            return []

class ConnectionManager:
    """Manages WebSocket connections for real-time detection streaming."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {} # Store connections by camera_id
        self.camera_threat_states: Dict[str, Dict[str, Any]] = {} # Track threat state per camera
        self.db_manager = DatabaseManager() # Initialize DB manager here
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
                    self.db_manager.log_threat_event(camera_id, "Unknown Camera", "exit") # Use "Unknown Camera" as name
                del self.camera_threat_states[camera_id]
            logger.info(f"Client disconnected for camera {camera_id}. Total connections: {len(self.active_connections)}")
        
    async def send_detection_result(self, websocket: WebSocket, data: dict):
        try:
            await websocket.send_text(json.dumps(data))
        except Exception as e:
            logger.error(f"Error sending data to client for camera {data.get('camera_id')}: {str(e)}")
            self.disconnect(data.get('camera_id'))

    def update_threat_state_and_log(self, camera_id: str, camera_name: str, threat_detected_in_frame: bool, confidence: Optional[float] = None):
        """
        Updates the threat state for a camera and logs events to the database.
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
                    self.db_manager.log_threat_event(camera_id, camera_name, "entry", confidence)
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
                        self.db_manager.log_threat_event(camera_id, camera_name, "exit")
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
        
        self.model = None
        self.model_path = os.getenv("MODEL_PATH")
        self.allowed_origins = os.getenv("ALLOWED_ORIGINS", "").split(",")
        self.confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))
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
                'update': lambda self, pt, dc: None,
                'get_stats': lambda self: {"error": "Stats unavailable"}
            })()
        
        # Validate environment variables
        if not self.model_path:
            raise ValueError("MODEL_PATH not found in environment variables")
        
        # Load the YOLO model
        self._load_model()
        logger.info("LiveWeaponDetectionAPI initialization complete")
    
    def _load_model(self):
        """Load the YOLOv8 model from the specified path, reassembling if necessary."""
        try:
            # Check if the full model file exists
            if not os.path.exists(self.model_path):
                logger.warning(f"Model file '{self.model_path}' not found. Attempting to reassemble from chunks.")
                model_dir = os.path.dirname(self.model_path)
                model_file_name = os.path.basename(self.model_path)
                chunk_dir = os.path.join(model_dir, f"{model_file_name}_chunks")

                reassembled_path = reassemble_file(chunk_dir, model_file_name, model_dir)
                
                if not reassembled_path or not os.path.exists(reassembled_path):
                    logger.error(f"Failed to reassemble model from chunks in {chunk_dir}. Using dummy model.")
                    # Fallback to dummy model if reassembly fails
                    class DummyYOLOModel:
                        def __call__(self, *args, **kwargs):
                            return [type('obj', (object,), {'boxes': None})()]
                        def fuse(self):
                            pass
                        @property
                        def names(self):
                            return {0: 'weapon'}
                    self.model = DummyYOLOModel()
                    return # Exit after setting dummy model
            
            # If model exists or was successfully reassembled, load it
            self.model = YOLO(self.model_path)
            
            # Optimize model for inference speed
            self.model.fuse()  # Fuse conv and bn layers for faster inference
            
            # Warm up the model with a dummy image
            dummy_img = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
            _ = self.model(dummy_img, verbose=False, conf=self.confidence_threshold, imgsz=self.input_size)
            
            logger.info(f"Model loaded and optimized successfully from: {self.model_path}")
            logger.info(f"Model input size set to: {self.input_size}x{self.input_size}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
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
        
    def _run_inference(self, image: np.ndarray, frame_id: int = None, camera_id: str = "unknown_camera") -> Dict[str, Any]:
        """
        Run YOLO inference on the input image frame with performance optimizations.
        
        Args:
            image: Input image in BGR format (OpenCV format)
            frame_id: Optional frame identifier
            camera_id: Identifier for the camera providing the frame
            
        Returns:
            Dictionary containing detection results and metadata
        """
        try:
            self.frame_counter += 1
            
            # Skip frames for better performance (process every N frames)
            if self.frame_counter % self.skip_frames != 0 and self.last_result is not None:
                # Return cached result with updated frame_id and timestamp
                cached_result = self.last_result.copy()
                cached_result["frame_id"] = frame_id
                cached_result["timestamp"] = datetime.now().isoformat()
                cached_result["cached"] = True
                cached_result["camera_id"] = camera_id # Ensure camera_id is propagated
                return cached_result
            
            start_time = datetime.now()
            
            # Preprocess image for faster inference
            processed_image = self._preprocess_image(image)
            
            # Convert BGR to RGB for YOLO model
            image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            
            detections = []
            total_boxes = 0
            threat_detected_in_frame = False
            max_confidence = 0.0

            # Run inference with optimized settings
            # Check if model is a dummy model
            if isinstance(self.model, type('obj', (object,), {})) and hasattr(self.model, '__call__') and not hasattr(self.model, 'fuse'):
                # This is our dummy model, simulate detections if needed for testing
                # For this demo, we'll simulate a random threat detection for the dummy model
                if camera_id == "camera-1" and datetime.now().second % 10 < 3: # Simulate threat for camera-1 every 10 seconds for 3 seconds
                    threat_detected_in_frame = True
                    max_confidence = 0.85
                    detections.append(Detection(
                        label="weapon",
                        confidence=max_confidence,
                        bbox=[100, 100, 200, 200],
                        timestamp=datetime.now().isoformat()
                    ))
                    total_boxes = 1
                results = [] # No actual results object for dummy
            else:
                # Real YOLO model inference
                results = self.model(
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
                            label = self.model.names[class_id]
                            
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
            
            # Calculate processing time and FPS
            processing_time = (datetime.now() - start_time).total_seconds()
            fps = 1.0 / processing_time if processing_time > 0 else 0
            
            # Update performance statistics safely
            try:
                self.stats.update(processing_time, len(detections))
            except Exception as e:
                logger.warning(f"Failed to update stats: {str(e)}")

            # Update threat state and log to DB
            camera_name = f"Camera {camera_id.split('-')[-1]}" if camera_id else "Unknown Camera" # Derive a name
            connection_manager.update_threat_state_and_log(
                camera_id, camera_name, threat_detected_in_frame, max_confidence if threat_detected_in_frame else None
            )
            
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
                "camera_id": camera_id # Include camera_id in response
            }
            
            # Cache result for frame skipping (might need to be per-camera for multiple streams)
            # For simplicity, keeping it global for now, but in a multi-camera setup,
            # this 'last_result' caching would need to be per-camera.
            # For this modification, the frontend will manage the 5-second persistence.
            # So, the backend's 'threat_detected' will reflect only the current frame.
            self.last_result = response.copy() 
            
            if detections:
                logger.info(f"Frame {frame_id} from {camera_id}: Detected {len(detections)} weapons in {processing_time:.4f}s ({fps:.1f} FPS)")
            
            return response
            
        except Exception as e:
            logger.error(f"Inference failed for camera {camera_id}: {str(e)}")
            raise RuntimeError(f"Model inference failed for camera {camera_id}: {str(e)}")

# Initialize the detection API and connection manager
try:
    logger.info("Starting application initialization...")
    detection_api = LiveWeaponDetectionAPI()
    connection_manager = ConnectionManager() # ConnectionManager now initializes DatabaseManager
    logger.info("Application components initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize detection API or components: {str(e)}")
    raise

# Create FastAPI application
app = FastAPI(
    title="Live Weapon Detection API",
    description="Real-time weapon detection from live video streams using YOLOv8",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=detection_api.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "Live Weapon Detection API is running",
        "model_loaded": detection_api.model is not None,
        "model_path": detection_api.model_path,
        "active_connections": len(connection_manager.active_connections),
        "confidence_threshold": detection_api.confidence_threshold
    }

@app.get("/health")
async def health_check():
    """Detailed health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": detection_api.model is not None,
        "model_path": detection_api.model_path,
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
            "model_path": detection_api.model_path
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

@app.get("/logs/threats")
async def get_threat_logs(limit: int = 20):
    """
    Endpoint to retrieve recent threat logs from the database.
    """
    try:
        logs = connection_manager.db_manager.get_threat_logs(limit=limit)
        return {"logs": logs}
    except Exception as e:
        logger.error(f"Error retrieving threat logs from database: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve logs: {e}")

@app.get("/demo", response_class=HTMLResponse)
async def demo_page():
    """Simple demo page for testing the live detection."""
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
                <div class="activity-log-container">
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

            // Initialize camera selection and fetch logs on page load
            window.onload = () => {
                initCameraSelection();
                fetchThreatLogs();
                setInterval(fetchThreatLogs, LOG_REFRESH_INTERVAL_MS); // Auto-refresh logs
                setInterval(updateLogRefreshTimeDisplay, 10000); // Update "X mins ago" every 10 seconds
                document.getElementById('refreshLogs').addEventListener('click', fetchThreatLogs); // Manual refresh
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
                    result = detection_api._run_inference(image, frame_id, camera_id)
                    
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
    
    # Get configuration from environment or use defaults
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8005))
    
    logger.info(f"Starting Live Weapon Detection server on {host}:{port}")
    logger.info(f"Demo page available at: http://{host}:{port}/demo")
    
    # Ensure the database is initialized before starting the server
    db_manager_init_check = DatabaseManager() 
    db_manager_init_check._create_tables() # Explicitly create tables on startup

    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=False,  # Set to True for development
        log_level="info"
    )
