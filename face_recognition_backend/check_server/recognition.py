# check_server/recognition.py
"""
FastAPI Face Recognition Server for Campus Face Recognition System
Handles real-time face recognition, access logging, and user status updates.
"""

import os
import cv2
import json
import time
import numpy as np
import face_recognition
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging
import base64

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import database and config utilities
from shared.database import (
    initialize_database, get_all_users, update_user_status, log_access_event, get_access_logs, get_user_by_id
)
from shared.config import (
    LOG_LEVEL, RECOGNITION_HOST, RECOGNITION_PORT,
    ENTRY_CAMERA_INDEX, EXIT_CAMERA_INDEX, CAMERA_FALLBACK_ENABLED,
    EMBEDDING_THRESHOLD, FACE_DETECTION_SCALE, COOLDOWN_PERIOD_SECONDS,
    CORS_ORIGINS, CORS_CREDENTIALS,
    UNKNOWN_COLOR, ROLE_COLORS # Assuming these are defined in config
)

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper()),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Campus Face Recognition API",
    description="API for real-time face recognition and access control",
    version="1.0.0"
)

# Add CORS middleware using values from config
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=CORS_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for known faces and webcams
known_face_encodings = []
known_face_names = []
known_user_ids = []
known_user_roles = []

entry_webcam: Optional[cv2.VideoCapture] = None
exit_webcam: Optional[cv2.VideoCapture] = None

last_entry_detection_time: Dict[str, float] = {}
last_exit_detection_time: Dict[str, float] = {}


# Utility functions
def load_known_faces() -> bool:
    """Loads known face encodings and names from the database."""
    global known_face_encodings, known_face_names, known_user_ids, known_user_roles
    
    known_face_encodings = []
    known_face_names = []
    known_user_ids = []
    known_user_roles = []

    users = get_all_users() # Use shared database utility
    
    if not users:
        logger.info("No users found in the database.")
        return True # Not an error, just no faces to load

    for user in users:
        try:
            # get_all_users now returns 'embedding' as a numpy array directly (or empty if failed)
            embedding = user.get('embedding')
            if embedding is not None and len(embedding) > 0:
                known_face_encodings.append(embedding)
                known_face_names.append(user['name'])
                known_user_ids.append(user['id_number'])
                known_user_roles.append(user['role'])
            else:
                logger.warning(f"User {user.get('id_number', 'N/A')} has no valid embedding.")
        except Exception as e:
            logger.error(f"Error processing user {user.get('id_number', 'N/A')} for known faces: {e}")
            
    logger.info(f"Loaded {len(known_face_encodings)} known faces from the database.")
    return True

def initialize_webcams():
    """Initializes entry and exit webcams."""
    global entry_webcam, exit_webcam

    # Entry Webcam
    logger.info(f"Attempting to initialize entry webcam (camera {ENTRY_CAMERA_INDEX}).")
    entry_webcam = cv2.VideoCapture(ENTRY_CAMERA_INDEX)
    if entry_webcam.isOpened():
        logger.info(f"Entry webcam (camera {ENTRY_CAMERA_INDEX}) initialized successfully.")
    else:
        logger.error(f"Could not initialize entry webcam (camera {ENTRY_CAMERA_INDEX}). Check connection or index.")
        entry_webcam = None

    # Exit Webcam
    logger.info(f"Attempting to initialize exit webcam (camera {EXIT_CAMERA_INDEX}).")
    exit_webcam = cv2.VideoCapture(EXIT_CAMERA_INDEX)
    if exit_webcam.isOpened():
        logger.info(f"Exit webcam (camera {EXIT_CAMERA_INDEX}) initialized successfully.")
    else:
        logger.warning(f"Could not initialize exit webcam (camera {EXIT_CAMERA_INDEX}). Check connection.")
        if CAMERA_FALLBACK_ENABLED and ENTRY_CAMERA_INDEX != EXIT_CAMERA_INDEX: # Prevent re-initializing same camera
            logger.info(f"Attempting to use entry camera index {ENTRY_CAMERA_INDEX} as fallback for exit webcam.")
            exit_webcam = cv2.VideoCapture(ENTRY_CAMERA_INDEX) # Fallback to entry camera
            if exit_webcam.isOpened():
                logger.info(f"Exit webcam initialized successfully using camera {ENTRY_CAMERA_INDEX} (fallback).")
            else:
                logger.critical(f"Failed to initialize exit webcam even with fallback. No exit webcam available.")
                exit_webcam = None # Still no exit webcam
        elif CAMERA_FALLBACK_ENABLED and ENTRY_CAMERA_INDEX == EXIT_CAMERA_INDEX:
            logger.info("Entry and Exit camera indices are the same, no fallback needed/possible.")
            # If they are the same, and entry is open, exit will also be handled by the same device.
            # If entry failed, exit will also fail. No further action needed here.
        else:
            logger.critical(f"Camera fallback is disabled, and exit webcam is not available.")
            exit_webcam = None # No exit webcam

# FastAPI Lifecycle Events
@app.on_event("startup")
async def startup_event():
    """Initializes database, loads known faces, and sets up webcams on startup."""
    logger.info("Recognition API starting up...")
    if not initialize_database():
        logger.critical("Failed to initialize database. Exiting.")
        raise Exception("Database initialization failed")

    if not load_known_faces():
        logger.critical("Failed to load known faces from database. Exiting.")
        raise Exception("Known faces loading failed")
    
    initialize_webcams()
    logger.info("Recognition API startup complete.")

@app.on_event("shutdown")
async def shutdown_event():
    """Releases webcam resources on shutdown."""
    logger.info("Recognition API shutting down...")
    if entry_webcam and entry_webcam.isOpened():
        entry_webcam.release()
        logger.info("Entry webcam released.")
    if exit_webcam and exit_webcam.isOpened():
        exit_webcam.release()
        logger.info("Exit webcam released.")
    logger.info("Recognition API shutdown complete.")

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint for the Recognition API."""
    return {
        "message": "Campus Face Recognition API",
        "version": "1.0.0",
        "endpoints": {
            "entry_stream": "/entry_stream",
            "exit_stream": "/exit_stream",
            "health": "/health",
            "logs": "/logs",
            "users": "/users/{user_id}",
            "reload_faces": "/reload-faces",
            "on_site_personnel": "/on_site_personnel", # Added
            "scanner_status": "/scanner_status" # Added
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "API is running"}

@app.get("/reload-faces")
async def reload_faces_endpoint():
    """Endpoint to manually reload known faces from the database."""
    logger.info("Manual reload of known faces initiated.")
    if load_known_faces():
        return {"status": "success", "message": f"Successfully reloaded {len(known_face_encodings)} known faces."}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload known faces.")

@app.get("/on_site_personnel")
async def get_on_site_personnel():
    """Retrieve a list of all personnel currently on-site."""
    try:
        all_users = get_all_users()
        on_site_users = [
            {"id_number": user['id_number'], "name": user['name'], "role": user['role']}
            for user in all_users if user.get('on_site')
        ]
        logger.info(f"Retrieved {len(on_site_users)} on-site personnel.")
        return {"on_site_personnel": on_site_users, "total": len(on_site_users)}
    except Exception as e:
        logger.error(f"Error retrieving on-site personnel: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve on-site personnel: {str(e)}")

@app.get("/scanner_status")
async def get_scanner_status():
    """Get the current status of the entry and exit scanners (webcams)."""
    entry_status = "active" if entry_webcam and entry_webcam.isOpened() else "inactive"
    exit_status = "active" if exit_webcam and exit_webcam.isOpened() else "inactive"
    
    logger.info(f"Scanner status requested: Entry={entry_status}, Exit={exit_status}")
    return {
        "entry_scanner": {"status": entry_status, "camera_index": ENTRY_CAMERA_INDEX},
        "exit_scanner": {"status": exit_status, "camera_index": EXIT_CAMERA_INDEX},
        "message": "Scanner status retrieved successfully."
    }


@app.websocket("/entry_stream")
async def entry_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Entry webcam WebSocket connected.")
    if not entry_webcam or not entry_webcam.isOpened():
        logger.error("Entry webcam not available for streaming.")
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR, reason="Entry webcam not initialized.")
        return

    try:
        while True:
            ret, frame = entry_webcam.read()
            if not ret:
                logger.error("Failed to read frame from entry webcam.")
                # Attempt to re-initialize camera if it failed to read
                initialize_webcams()
                if not entry_webcam or not entry_webcam.isOpened():
                    logger.critical("Entry webcam permanently unavailable. Closing stream.")
                    break
                continue # Try reading again after re-init

            # Process frame for face recognition
            small_frame = cv2.resize(frame, (0, 0), fx=FACE_DETECTION_SCALE, fy=FACE_DETECTION_SCALE)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            current_faces_info = []

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Scale back up face locations
                top = int(top / FACE_DETECTION_SCALE)
                right = int(right / FACE_DETECTION_SCALE)
                bottom = int(bottom / FACE_DETECTION_SCALE)
                left = int(left / FACE_DETECTION_SCALE)

                name = "Unknown"
                user_id = None
                role = None
                confidence = 0.0
                box_color = UNKNOWN_COLOR # Default to unknown color

                if known_face_encodings:
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    matches = face_distances <= EMBEDDING_THRESHOLD

                    if True in matches:
                        first_match_index = np.where(matches)[0][0]
                        name = known_face_names[first_match_index]
                        user_id = known_user_ids[first_match_index]
                        role = known_user_roles[first_match_index]
                        confidence = 1 - face_distances[first_match_index] # Convert distance to confidence

                        # Set color based on role if known
                        box_color = ROLE_COLORS.get(role, UNKNOWN_COLOR)


                        current_time = time.time()
                        if user_id and (user_id not in last_entry_detection_time or
                                        current_time - last_entry_detection_time[user_id] > COOLDOWN_PERIOD_SECONDS):
                            
                            logger.info(f"User {name} (ID: {user_id}, Role: {role}) detected for entry. Confidence: {confidence:.2f}")
                            if update_user_status(user_id, True): # Set on_site to True
                                log_access_event(user_id, name, 'entry', 'scanner', confidence)
                                last_entry_detection_time[user_id] = current_time
                            else:
                                logger.error(f"Failed to update status for user {user_id}")
                                

                current_faces_info.append({
                    "name": name,
                    "confidence": f"{confidence:.2f}",
                    "location": [top, right, bottom, left],
                    "action": "entry"
                })
                # Draw box and label
                cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), box_color, cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                label = f"{name} ({confidence:.2f})" if name != "Unknown" else "Unknown"
                cv2.putText(frame, label, (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 1)

            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ret:
                logger.error("Failed to encode frame to JPEG for entry stream.")
                continue
            
            # Send image data and detection info
            await websocket.send_json({
                "image": base64.b64encode(buffer).decode('utf-8'),
                "detections": current_faces_info
            })
            await websocket.send_text("FRAME_END") # Signal end of frame data

            # Small delay to control frame rate if necessary, derived from STREAM_FPS
            # This aims for roughly STREAM_FPS frames per second.
            # However, face recognition is CPU intensive, so actual FPS will be lower.
            # time.sleep(1.0 / STREAM_FPS) 

    except WebSocketDisconnect:
        logger.info("Entry webcam WebSocket disconnected.")
    except Exception as e:
        logger.error(f"Error in entry webcam WebSocket: {e}", exc_info=True)
    finally:
        if websocket.client_state == status.WS_CONNECTED:
            await websocket.close()


@app.websocket("/exit_stream")
async def exit_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Exit webcam WebSocket connected.")
    if not exit_webcam or not exit_webcam.isOpened():
        logger.error("Exit webcam not available for streaming.")
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR, reason="Exit webcam not initialized.")
        return

    try:
        while True:
            ret, frame = exit_webcam.read()
            if not ret:
                logger.error("Failed to read frame from exit webcam.")
                # Attempt to re-initialize camera if it failed to read
                initialize_webcams()
                if not exit_webcam or not exit_webcam.isOpened():
                    logger.critical("Exit webcam permanently unavailable. Closing stream.")
                    break
                continue # Try reading again after re-init

            # Process frame for face recognition
            small_frame = cv2.resize(frame, (0, 0), fx=FACE_DETECTION_SCALE, fy=FACE_DETECTION_SCALE)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            current_faces_info = []

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Scale back up face locations
                top = int(top / FACE_DETECTION_SCALE)
                right = int(right / FACE_DETECTION_SCALE)
                bottom = int(bottom / FACE_DETECTION_SCALE)
                left = int(left / FACE_DETECTION_SCALE)

                name = "Unknown"
                user_id = None
                role = None
                confidence = 0.0
                box_color = UNKNOWN_COLOR # Default to unknown color

                if known_face_encodings:
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    matches = face_distances <= EMBEDDING_THRESHOLD

                    if True in matches:
                        first_match_index = np.where(matches)[0][0]
                        name = known_face_names[first_match_index]
                        user_id = known_user_ids[first_match_index]
                        role = known_user_roles[first_match_index]
                        confidence = 1 - face_distances[first_match_index] # Convert distance to confidence

                        # Set color based on role if known
                        box_color = ROLE_COLORS.get(role, UNKNOWN_COLOR)

                        current_time = time.time()
                        if user_id and (user_id not in last_exit_detection_time or
                                        current_time - last_exit_detection_time[user_id] > COOLDOWN_PERIOD_SECONDS):
                            
                            logger.info(f"User {name} (ID: {user_id}, Role: {role}) detected for exit. Confidence: {confidence:.2f}")
                            if update_user_status(user_id, False): # Set on_site to False
                                log_access_event(user_id, name, 'exit', 'scanner', confidence)
                                last_exit_detection_time[user_id] = current_time
                            else:
                                logger.error(f"Failed to update status for user {user_id}")

                current_faces_info.append({
                    "name": name,
                    "confidence": f"{confidence:.2f}",
                    "location": [top, right, bottom, left],
                    "action": "exit"
                })
                # Draw box and label
                cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), box_color, cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                label = f"{name} ({confidence:.2f})" if name != "Unknown" else "Unknown"
                cv2.putText(frame, label, (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 1)

            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ret:
                logger.error("Failed to encode frame to JPEG for exit stream.")
                continue
            
            # Send image data and detection info
            await websocket.send_json({
                "image": base64.b64encode(buffer).decode('utf-8'),
                "detections": current_faces_info
            })
            await websocket.send_text("FRAME_END") # Signal end of frame data

            # Small delay to control frame rate if necessary
            # time.sleep(1.0 / STREAM_FPS)

    except WebSocketDisconnect:
        logger.info("Exit webcam WebSocket disconnected.")
    except Exception as e:
        logger.error(f"Error in exit webcam WebSocket: {e}", exc_info=True)
    finally:
        if websocket.client_state == status.WS_CONNECTED:
            await websocket.close()


@app.get("/logs")
async def get_logs_endpoint(user_id: Optional[str] = None, action: Optional[str] = None, limit: int = 100):
    """Retrieve access logs from the database."""
    try:
        logs = get_access_logs(user_id=user_id, action=action, limit=limit)
        return {"logs": logs, "total": len(logs)}
    except Exception as e:
        logger.error(f"Error retrieving logs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve logs: {str(e)}")

@app.get("/users/{user_id}/status")
async def get_user_status(user_id: str):
    """Get a specific user's current on-site status."""
    user_data = get_user_by_id(user_id)
    if user_data:
        return {"user_id": user_id, "on_site": user_data['on_site']}
    else:
        raise HTTPException(status_code=404, detail="User not found")

@app.post("/users/{user_id}/manual_entry")
async def manual_entry(user_id: str):
    """Manually log a user entry."""
    user_data = get_user_by_id(user_id)
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")
    
    if update_user_status(user_id, True):
        log_access_event(user_id, user_data['name'], 'entry', 'manual')
        return {"status": "success", "message": f"Manual entry logged for {user_data['name']}."}
    else:
        raise HTTPException(status_code=500, detail="Failed to update user status for manual entry.")

@app.post("/users/{user_id}/manual_exit")
async def manual_exit(user_id: str):
    """Manually log a user exit."""
    user_data = get_user_by_id(user_id)
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")
    
    if update_user_status(user_id, False):
        log_access_event(user_id, user_data['name'], 'exit', 'manual')
        return {"status": "success", "message": f"Manual exit logged for {user_data['name']}."}
    else:
        raise HTTPException(status_code=500, detail="Failed to update user status for manual exit.")

if __name__ == "__main__":
    uvicorn.run(app, host=RECOGNITION_HOST, port=RECOGNITION_PORT, log_level=LOG_LEVEL.lower())