# Live Weapon Detection API Documentation

## **Core Classes**

### **Performance Monitoring**
- **`PerformanceStats`** - Tracks total frames processed, detections made, processing times, and calculates average FPS and uptime for monitoring system performance

### **Database Management**
- **`DatabaseManager`** - Manages SQLite database operations for threat event logging, creates tables, logs entry/exit events with timestamps and confidence scores
- **`DatabaseManager.log_threat_event(camera_id, camera_name, action, confidence)`** - Records threat events ("entry"/"exit") with camera details and optional confidence scores
- **`DatabaseManager.get_threat_logs(limit)`** - Retrieves recent threat logs from database, returns list of dictionaries with event details

### **Connection Management**
- **`ConnectionManager`** - Manages WebSocket connections for real-time detection streaming across multiple cameras
- **`ConnectionManager.connect(websocket, camera_id)`** - Establishes WebSocket connection and initializes threat state tracking for specific camera
- **`ConnectionManager.update_threat_state_and_log(camera_id, camera_name, threat_detected, confidence)`** - Updates threat persistence state (5-second window) and logs entry/exit events to database

### **Main Detection Engine**
- **`LiveWeaponDetectionAPI`** - Core weapon detection system using YOLOv8 model for real-time threat detection from video streams

---

## **Core Functions**

### **Model Management**
- **`_load_model()`** - Loads YOLOv8 model from specified path, attempts reassembly from chunks if needed, optimizes model with layer fusion and warm-up
- **`chunk_file(file_path, chunk_size_mb)`** - Splits large model files into smaller chunks for storage/transfer (from chunker module)
- **`reassemble_file(chunk_dir, filename, output_dir)`** - Reconstructs original model file from chunks (from collector module)

### **Image Processing Functions**
- **`_decode_base64_image(base64_string)`** - Converts base64 encoded image string to OpenCV BGR format for processing
- **`_preprocess_image(image)`** - Resizes images to model input size while maintaining aspect ratio for faster inference
- **`_run_inference(image, frame_id, camera_id)`** - Executes YOLO detection on processed image, applies confidence filtering, updates threat states, returns detection results

### **Performance Optimization**
- **Frame skipping** - Processes every N frames (configurable via SKIP_FRAMES) and caches results for intermediate frames
- **Input size optimization** - Configurable input resolution (default 416x416) for speed vs accuracy balance
- **Model optimization** - Layer fusion, FP16 precision, and model warm-up for faster inference

---

## **API Endpoints**

### **System Health Endpoints**
- **`GET /`** - **Root health check** returning API status, model loading state, active connections, and configuration
- **`GET /health`** - **Detailed health check** with comprehensive system status, configuration parameters, and connection details
- **`GET /stats`** - **Performance statistics** including FPS, processing times, total frames processed, and uptime metrics

### **Model Management Endpoints**
- **`POST /model/chunk`** - **Chunk model files** for storage optimization, accepts list of model paths and chunk size configuration
- **`GET /logs/threats`** - **Retrieve threat logs** from database with configurable limit, returns recent entry/exit events with timestamps

### **Real-time Detection**
- **`WebSocket /ws/detect?camera_id={id}`** - **Live detection stream** accepting base64 video frames, returns real-time detection results with bounding boxes and confidence scores

### **Demo Interface**
- **`GET /demo`** - **Interactive demo page** with multi-camera support, real-time threat visualization, activity logging, and popup notifications

---

## **WebSocket Message Format**

### **Client to Server (Frame Data)**
```json
{
    "type": "frame",
    "data": "base64_encoded_image_data",
    "frame_id": 12345,
    "camera_id": "camera-1",
    "camera_name": "Front Entrance Camera"
}
```

### **Server to Client (Detection Results)**
```json
{
    "detections": [
        {
            "label": "weapon",
            "confidence": 0.85,
            "bbox": [x1, y1, x2, y2],
            "timestamp": "2025-07-25T10:30:45"
        }
    ],
    "frame_id": 12345,
    "processing_time": 0.0234,
    "fps": 42.7,
    "threat_detected": true,
    "camera_id": "camera-1"
}
```

---

## **Key Features**

### **Multi-Camera Support**
- **Simultaneous processing** of multiple camera feeds with individual WebSocket connections
- **Per-camera threat state tracking** with 5-second persistence window
- **Camera-specific logging** with unique identifiers and friendly names

### **Threat Detection & Logging**
- **Real-time weapon detection** using YOLOv8 model with configurable confidence thresholds
- **Automatic threat logging** with entry/exit event tracking and timestamps
- **Threat persistence** (5-second window) to prevent flickering alerts
- **Database storage** of all threat events with camera details and confidence scores

### **Performance Optimization**
- **Frame rate control** (configurable FPS limiting for processing efficiency)
- **Dynamic frame skipping** with result caching for smoother performance
- **Configurable input resolution** for speed vs accuracy tuning
- **Model optimization** with layer fusion and FP16 precision

### **Web Interface Features**
- **Multi-camera dashboard** with real-time video feeds and detection overlays
- **Live activity log** showing recent threat entry/exit events
- **Visual threat alerts** with popup notifications and camera highlighting
- **Performance monitoring** displaying FPS, processing times, and connection status

### **Configuration Options**
- **Environment variables** for model path, confidence threshold, max detections, input size
- **CORS support** with configurable allowed origins
- **Flexible deployment** with Docker-ready configuration and health monitoring