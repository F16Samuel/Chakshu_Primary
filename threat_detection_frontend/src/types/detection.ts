// src/types/detection.ts

// Represents a single detection object from the backend
export interface Detection {
  label: string;
  confidence: number;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2]
  timestamp: string; // ISO 8601 string
}

// Represents the full detection result object from the WebSocket
export interface DetectionResult {
  detections: Detection[];
  frame_id?: number; // Corrected: Optional number
  processing_time: number;
  fps: number;
  timestamp: string; // ISO 8601 string
  threat_detected: boolean; // Indicates if any threat was detected in this frame
  total_boxes_detected: number;
  boxes_after_filtering: number;
  cached: boolean; // Added: Present in backend response
  frame_counter: number; // Added: Present in backend response
  camera_id: string;
}

// Represents the status of a camera feed in the frontend
export interface CameraStatus {
  id: string;
  deviceId: string; // Added: Crucial for camera selection
  name: string;
  status: 'initializing' | 'connected' | 'disconnected' | 'error';
  lastDetection: DetectionResult | null;
  threatActive: boolean; // Frontend state for persistent threat highlighting
  threatTimeout?: NodeJS.Timeout; // To clear the threat highlighting timeout
}

// Represents a single threat log entry from the backend's /logs/threats endpoint
export interface ThreatLog {
  id: number; // Primary key from SQLite
  camera_id: string;
  camera_name: string;
  action: 'entry' | 'exit'; // 'entry' or 'exit'
  timestamp: string; // ISO 8601 format
  method?: string; // Added: 'scanner' or 'manual' - Optional field from backend
  confidence?: number; // Optional: for scanner entries
}

// Represents the structure of the backend's /stats endpoint response
export interface AppStats {
  status: string;
  performance: { // This is the crucial nested object
    total_frames_processed: number;
    total_detections: number;
    average_fps: number;
    uptime_seconds: number;
    average_processing_time: number;
  };
  configuration: {
    confidence_threshold: number;
    max_detections: number;
    model_path: string;
  };
  connections: {
    active_connections: number;
  };
}
