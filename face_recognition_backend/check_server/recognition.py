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
from typing import Dict, Any, List, Optional, Tuple
import logging
import base64

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, status, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import database and config utilities
from shared.database import (
    initialize_database, get_all_users, update_user_status, log_access_event, get_access_logs, get_user_by_id
)
from shared.config import (
    LOG_LEVEL, RECOGNITION_HOST, RECOGNITION_PORT,
    # Removed camera indices as live streams are gone
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
    description="API for real-time face recognition and access control for kiosk system",
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

# Global variables for known faces
known_face_encodings = []
known_face_names = []
known_user_ids = []
known_user_roles = []

# No longer need webcam objects
# entry_webcam: Optional[cv2.VideoCapture] = None
# exit_webcam: Optional[cv2.VideoCapture] = None

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

# Removed initialize_webcams as webcams are no longer used for streaming

def process_static_image_recognition(
    base64_image: str, 
    detection_type: str, 
    last_detection_times: Dict[str, float]
) -> Dict[str, Any]:
    """
    Processes a single base64 encoded image for face recognition,
    updates user status, and logs events.
    Args:
        base64_image: The base64 encoded image string.
        detection_type: 'entry' or 'exit'
        last_detection_times: A dictionary to store the last detection time for each user.
    Returns:
        A dictionary containing recognition results.
    """
    try:
        # Decode the base64 image
        img_bytes = base64.b64decode(base64_image)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            logger.error("Failed to decode image from base64 string.")
            return {"status": "error", "message": "Invalid image data provided."}

        small_frame = cv2.resize(frame, (0, 0), fx=FACE_DETECTION_SCALE, fy=FACE_DETECTION_SCALE)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        if not face_encodings:
            return {"status": "no_face_detected", "message": "No face detected in the image."}

        # Assuming only one face for kiosk system, or just processing the first one found
        face_encoding = face_encodings[0]
        
        name = "Unknown"
        user_id = None
        role = None
        confidence = 0.0

        if known_face_encodings:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            matches = face_distances <= EMBEDDING_THRESHOLD

            if True in matches:
                first_match_index = np.where(matches)[0][0]
                name = known_face_names[first_match_index]
                user_id = known_user_ids[first_match_index]
                role = known_user_roles[first_match_index]
                confidence = 1 - face_distances[first_match_index] # Convert distance to confidence

                current_time = time.time()
                if user_id and (user_id not in last_detection_times or
                                current_time - last_detection_times[user_id] > COOLDOWN_PERIOD_SECONDS):
                    
                    logger.info(f"User {name} (ID: {user_id}, Role: {role}) detected for {detection_type}. Confidence: {confidence:.2f}")
                    
                    # Update user status based on detection_type
                    is_on_site = True if detection_type == 'entry' else False
                    if update_user_status(user_id, is_on_site):
                        log_access_event(user_id, name, detection_type, 'kiosk', confidence)
                        last_detection_times[user_id] = current_time
                        return {
                            "status": "success",
                            "message": f"User {name} recognized and {detection_type} logged.",
                            "user_id": user_id,
                            "name": name,
                            "role": role,
                            "confidence": f"{confidence:.2f}",
                            "action": detection_type,
                            "on_site_status_updated": True
                        }
                    else:
                        logger.error(f"Failed to update status for user {user_id}")
                        return {
                            "status": "error",
                            "message": f"Failed to update status for user {name} (ID: {user_id}).",
                            "user_id": user_id,
                            "name": name,
                            "role": role,
                            "confidence": f"{confidence:.2f}",
                            "action": detection_type,
                            "on_site_status_updated": False
                        }
                else:
                    # User recognized but within cooldown period
                    return {
                        "status": "cooldown",
                        "message": f"User {name} recognized, but within cooldown period for {detection_type}.",
                        "user_id": user_id,
                        "name": name,
                        "role": role,
                        "confidence": f"{confidence:.2f}",
                        "action": detection_type,
                        "on_site_status_updated": False
                    }
            else:
                # No match found within threshold
                return {
                    "status": "unknown_face",
                    "message": "Face detected, but not recognized as a known user.",
                    "confidence": "N/A",
                    "action": detection_type
                }
        else:
            # No known faces loaded in the system
            return {
                "status": "no_known_faces",
                "message": "No known faces are loaded in the system for recognition.",
                "confidence": "N/A",
                "action": detection_type
            }

    except Exception as e:
        logger.error(f"Error in process_static_image_recognition for {detection_type}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during recognition: {str(e)}")


# FastAPI Lifecycle Events
@app.on_event("startup")
async def startup_event():
    """Initializes database and loads known faces on startup."""
    logger.info("Recognition API starting up...")
    if not initialize_database():
        logger.critical("Failed to initialize database. Exiting.")
        raise Exception("Database initialization failed")

    if not load_known_faces():
        logger.critical("Failed to load known faces from database. Exiting.")
        raise Exception("Known faces loading failed")
    
    # Removed webcam initialization
    logger.info("Recognition API startup complete.")

@app.on_event("shutdown")
async def shutdown_event():
    """No webcam resources to release on shutdown in kiosk mode."""
    logger.info("Recognition API shutting down...")
    # No webcams to release
    logger.info("Recognition API shutdown complete.")

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint for the Recognition API."""
    return {
        "message": "Campus Face Recognition API (Kiosk Mode)",
        "version": "1.0.0",
        "endpoints": {
            "enter_site_recognition": "/enter_site_recognition (POST)",
            "exit_site_recognition": "/exit_site_recognition (POST)",
            "health": "/health",
            "logs": "/logs",
            "users_status": "/users/{user_id}/status",
            "on_site_personnel": "/on_site_personnel",
            "reload_faces": "/reload-faces",
            "manual_entry": "/users/{user_id}/manual_entry (POST)",
            "manual_exit": "/users/{user_id}/manual_exit (POST)"
            # Removed scanner_status as there are no 'scanners' in the old sense
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

# Removed get_scanner_status as it's no longer relevant for kiosk system

@app.post("/enter_site_recognition")
async def enter_site_recognition_endpoint(request: Request):
    """
    Endpoint for entering a site via face recognition.
    Expects a JSON body with 'base64_image'.
    """
    try:
        data = await request.json()
        base64_image = data.get("base64_image")
        if not base64_image:
            raise HTTPException(status_code=400, detail="No base64_image provided in the request body.")
        
        return process_static_image_recognition(base64_image, 'entry', last_entry_detection_time)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format.")
    except Exception as e:
        logger.error(f"Error in /enter_site_recognition endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing entry recognition: {str(e)}")

@app.post("/exit_site_recognition")
async def exit_site_recognition_endpoint(request: Request):
    """
    Endpoint for exiting a site via face recognition.
    Expects a JSON body with 'base64_image'.
    """
    try:
        data = await request.json()
        base64_image = data.get("base64_image")
        if not base64_image:
            raise HTTPException(status_code=400, detail="No base64_image provided in the request body.")

        return process_static_image_recognition(base64_image, 'exit', last_exit_detection_time)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format.")
    except Exception as e:
        logger.error(f"Error in /exit_site_recognition endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing exit recognition: {str(e)}")


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