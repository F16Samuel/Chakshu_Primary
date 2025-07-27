# main.py - Unified FastAPI Server for Campus Face Recognition System

import os
import json
import shutil
import numpy as np
import face_recognition
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import base64
import logging
from datetime import datetime, timedelta
import asyncio
import cv2
import time

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from mongo_database import MongoDB

# Load environment variables from .env file
load_dotenv()

# --- Configuration from .env ---
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
CARDS_DIR = Path(os.getenv("CARDS_DIR", "./uploads/cards"))
FACES_DIR = Path(os.getenv("FACES_DIR", "./uploads/faces"))
EMBEDDING_THRESHOLD = float(os.getenv("EMBEDDING_THRESHOLD", 0.6))
RECOGNITION_FPS = int(os.getenv("RECOGNITION_FPS", 2))
STREAM_FPS = int(os.getenv("STREAM_FPS", 60))
FACE_DETECTION_SCALE = float(os.getenv("FACE_DETECTION_SCALE", 0.25))
COOLDOWN_PERIOD_SECONDS = int(os.getenv("COOLDOWN_PERIOD_SECONDS", 5))
REQUIRE_WEBCAM_PHOTO = os.getenv("REQUIRE_WEBCAM_PHOTO", "false").lower() == 'true'
MIN_FACE_PHOTOS = int(os.getenv("MIN_FACE_PHOTOS", 2))
MAX_FACE_PHOTOS = int(os.getenv("MAX_FACE_PHOTOS", 5)) # Corrected typo: MAX_FILE_PHOTOS to MAX_FACE_PHOTOS
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 10485760))
ALLOWED_MIME_TYPES = os.getenv("ALLOWED_MIME_TYPES", "image/jpeg,image/jpg,image/png").split(',')
ENTRY_CAMERA_INDEX = int(os.getenv("ENTRY_CAMERA_INDEX", 0))
EXIT_CAMERA_INDEX = int(os.getenv("EXIT_CAMERA_INDEX", 1))
CAMERA_FALLBACK_ENABLED = os.getenv("CAMERA_FALLBACK_ENABLED", "true").lower() == 'true'
FACE_PROCESSING_INTERVAL = float(os.getenv("FACE_PROCESSING_INTERVAL", 0.5))
FRAME_TIMEOUT = int(os.getenv("FRAME_TIMEOUT", 30))
DEFAULT_ENTRY_SCANNING = os.getenv("DEFAULT_ENTRY_SCANNING", "false").lower() == 'true'
DEFAULT_EXIT_SCANNING = os.getenv("DEFAULT_EXIT_SCANNING", "false").lower() == 'true'
CAMERA_RETRY_ATTEMPTS = int(os.getenv("CAMERA_RETRY_ATTEMPTS", 3))
CAMERA_RETRY_DELAY = int(os.getenv("CAMERA_RETRY_DELAY", 2))
MAX_CONCURRENT_STREAMS = int(os.getenv("MAX_CONCURRENT_STREAMS", 2))
MEMORY_CLEANUP_INTERVAL = int(os.getenv("MEMORY_CLEANUP_INTERVAL", 300))
API_TITLE = os.getenv("API_TITLE", "Campus Face Recognition & Registration System")
API_DESCRIPTION = os.getenv("API_DESCRIPTION", "Unified API for user management, face recognition, and access control.")
API_VERSION = os.getenv("API_VERSION", "1.0.0")
ENABLE_RATE_LIMITING = os.getenv("ENABLE_RATE_LIMITING", "true").lower() == 'true'
MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", 60))

# Corrected CORS_ORIGINS parsing
cors_origins_env = os.getenv("CORS_ORIGINS", "*")
if cors_origins_env == "*":
    CORS_ORIGINS = ["*"]
else:
    CORS_ORIGINS = [origin.strip() for origin in cors_origins_env.split(',')]

CORS_CREDENTIALS = os.getenv("CORS_CREDENTIALS", "true").lower() == 'true'
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Parse UNKNOWN_COLOR and ROLE_COLORS
def parse_color_tuple(s: str) -> Tuple[int, int, int]:
    return tuple(map(int, s.strip('()').split(',')))

UNKNOWN_COLOR = parse_color_tuple(os.getenv("UNKNOWN_COLOR", "(0, 165, 255)"))
ROLE_COLORS_STR = os.getenv("ROLE_COLORS", '{"student": [0, 255, 0], "professor": [255, 0, 0], "guard": [255, 0, 255], "maintenance": [0, 255, 255]}')
ROLE_COLORS = json.loads(ROLE_COLORS_STR) # This assumes ROLE_COLORS is a valid JSON string
VALID_ROLES = json.loads(os.getenv("VALID_ROLES", '["student", "professor", "guard", "maintenance"]'))


# --- Configure logging ---
logging.basicConfig(level=getattr(logging, LOG_LEVEL),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Ensure directories exist ---
CARDS_DIR.mkdir(parents=True, exist_ok=True)
FACES_DIR.mkdir(parents=True, exist_ok=True)

# --- Import MongoDB database functions ---
from mongo_database import (
    get_mongo_db, save_user, check_user_exists, get_user_by_id, get_all_users,
    update_user_status, log_access_event, get_access_logs,
    get_dashboard_activities, get_personnel_breakdown, get_total_on_site, get_today_stats
)

# --- Import face processing and validation utilities ---
from face_processing import (
    extract_face_encoding, extract_face_encodings_from_bytes,
    compare_faces, generate_average_embedding, calculate_confidence,
    draw_face_bounding_box
)
from validation import (
    validate_file_upload, validate_image_quality, validate_user_data,
    validate_user_role, validate_id_number
)

# --- FastAPI App Initialization ---
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=CORS_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global State for Recognition ---
known_face_encodings = []
known_face_ids = []
last_recognition_time = {} # {user_id: datetime_object}
camera_streams = {} # {camera_index: cv2.VideoCapture object}
scanner_status = {
    "entry_scanning": DEFAULT_ENTRY_SCANNING,
    "exit_scanning": DEFAULT_EXIT_SCANNING
}

# --- Utility Functions ---

async def load_known_faces():
    """Loads all known face embeddings from the database into memory."""
    global known_face_encodings, known_face_ids
    logger.info("Loading known faces from database...")
    users = get_all_users()
    known_face_encodings = []
    known_face_ids = []
    for user in users:
        if 'face_embedding' in user and user['face_embedding'] is not None:
            # Ensure it's a numpy array for face_recognition library
            embedding = np.array(user['face_embedding'])
            if embedding.shape == (128,): # Ensure it's a valid 128-dim embedding
                known_face_encodings.append(embedding)
                known_face_ids.append(user['id_number'])
            else:
                logger.warning(f"Skipping invalid embedding for user {user['id_number']}: {embedding.shape}")
        else:
            logger.warning(f"User {user['id_number']} has no face embedding.")
    logger.info(f"Loaded {len(known_face_ids)} known faces.")

# Call this on startup
@app.on_event("startup")
async def startup_event():
    logger.info("Starting up FastAPI application...")
    # Ensure MongoDB connection is established
    try:
        get_mongo_db()
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB on startup: {e}")
        # Depending on criticality, you might want to exit here or retry
    await load_known_faces()
    logger.info("FastAPI application started.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down FastAPI application...")
    # This assumes MongoDB() is a class that needs to be instantiated to call close_connection
    # If get_mongo_db() returns an instance, you should use that instance to close.
    # For now, I'll assume get_mongo_db() returns a client or a way to get it.
    # If get_mongo_db() returns the client directly, you'd need something like:
    # try:
    #     mongo_client = get_mongo_db().client # Assuming get_mongo_db returns a DB object with a .client attribute
    #     mongo_client.close()
    #     logger.info("MongoDB connection closed.")
    # except Exception as e:
    #     logger.error(f"Error closing MongoDB connection: {e}")

    # Release camera resources
    for cam_index, cap in camera_streams.items():
        if cap and cap.isOpened():
            cap.release()
            logger.info(f"Released camera {cam_index}")
    logger.info("FastAPI application shut down.")

# --- Models ---
class UserRegistration(BaseModel):
    id_number: str
    name: str
    role: str
    # webcam_photo_data: Optional[str] = None # Base64 encoded image

class ManualAccess(BaseModel):
    user_id: str

# --- Endpoints: Health Check ---
@app.get("/health")
async def health_check():
    """Checks the health of the API."""
    try:
        # Attempt to ping MongoDB to check database health
        get_mongo_db().command('ping')
        return JSONResponse(content={"status": "ok", "database": "connected"}, status_code=200)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(content={"status": "error", "database": "disconnected", "detail": str(e)}, status_code=500)

# --- Endpoints: User Registration (from reg_server/registration.py) ---

@app.post("/register")
async def register_user(
    id_number: str = Form(...),
    name: str = Form(...),
    role: str = Form(...),
    aadhar_card: UploadFile = File(None),
    role_id_card: UploadFile = File(None),
    face_photo_1: UploadFile = File(None),
    face_photo_2: UploadFile = File(None)
):
    """
    Registers a new user with their details and face encoding.
    Requires at least one face photo (webcam or uploaded).
    """
    logger.info(f"Attempting to register user: {name} ({id_number}) with role: {role}")

    # 1. Validate input data, removed valid_roles as it's not needed by validate_user_data directly
    validate_user_data(name, role, id_number)

    if await check_user_exists(id_number):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"User with ID number '{id_number}' already exists.")

    user_face_encodings = []
    face_photo_paths = []

    # Process face photo 1 if provided
    if face_photo_1 and face_photo_1.filename:
        logger.info(f"Processing face_photo_1 for {id_number}")
        photo_bytes = await face_photo_1.read()
        # Pass ALLOWED_MIME_TYPES and MAX_FILE_SIZE from main.py's config
        validate_file_upload(face_photo_1, allowed_types=ALLOWED_MIME_TYPES, max_size=MAX_FILE_SIZE)
        if not validate_image_quality(photo_bytes):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Face photo 1 quality is insufficient.")

        encoding = extract_face_encodings_from_bytes(photo_bytes)
        if encoding:
            user_face_encodings.extend(encoding)
            photo_path = FACES_DIR / f"{id_number}_face_1_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            with open(photo_path, "wb") as buffer:
                buffer.write(photo_bytes)
            face_photo_paths.append(str(photo_path))
        else:
            logger.warning(f"No face found in face_photo_1 for {id_number}.")

    # Process face photo 2 if provided
    if face_photo_2 and face_photo_2.filename:
        logger.info(f"Processing face_photo_2 for {id_number}")
        photo_bytes = await face_photo_2.read()
        # Pass ALLOWED_MIME_TYPES and MAX_FILE_SIZE from main.py's config
        validate_file_upload(face_photo_2, allowed_types=ALLOWED_MIME_TYPES, max_size=MAX_FILE_SIZE)
        if not validate_image_quality(photo_bytes):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Face photo 2 quality is insufficient.")

        encoding = extract_face_encodings_from_bytes(photo_bytes)
        if encoding:
            user_face_encodings.extend(encoding)
            photo_path = FACES_DIR / f"{id_number}_face_2_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            with open(photo_path, "wb") as buffer:
                buffer.write(photo_bytes)
            face_photo_paths.append(str(photo_path))
        else:
            logger.warning(f"No face found in face_photo_2 for {id_number}.")


    # Process Aadhar card if provided
    aadhar_path = None
    if aadhar_card and aadhar_card.filename:
        logger.info(f"Processing Aadhar card for {id_number}")
        # Pass ALLOWED_MIME_TYPES and MAX_FILE_SIZE from main.py's config
        validate_file_upload(aadhar_card, allowed_types=ALLOWED_MIME_TYPES, max_size=MAX_FILE_SIZE)
        aadhar_path = CARDS_DIR / f"{id_number}_aadhar.{aadhar_card.filename.split('.')[-1]}"
        try:
            with open(aadhar_path, "wb") as buffer:
                shutil.copyfileobj(aadhar_card.file, buffer)
            logger.info(f"Aadhar card saved to {aadhar_path}")
        except Exception as e:
            logger.error(f"Failed to save Aadhar card: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to save Aadhar card.")

    # Process Role ID card if provided
    role_id_path = None
    if role_id_card and role_id_card.filename:
        logger.info(f"Processing Role ID card for {id_number}")
        # Pass ALLOWED_MIME_TYPES and MAX_FILE_SIZE from main.py's config
        validate_file_upload(role_id_card, allowed_types=ALLOWED_MIME_TYPES, max_size=MAX_FILE_SIZE)
        role_id_path = CARDS_DIR / f"{id_number}_role_id.{role_id_card.filename.split('.')[-1]}"
        try:
            with open(role_id_path, "wb") as buffer:
                shutil.copyfileobj(role_id_card.file, buffer)
            logger.info(f"Role ID card saved to {role_id_path}")
        except Exception as e:
            logger.error(f"Failed to save Role ID card: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to save Role ID card.")

    # Ensure at least MIN_FACE_PHOTOS are provided for face recognition
    if len(user_face_encodings) < MIN_FACE_PHOTOS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"At least {MIN_FACE_PHOTOS} face photo(s) are required for registration. Only {len(user_face_encodings)} faces detected in provided photos."
        )

    # Generate average embedding
    average_embedding = generate_average_embedding(user_face_encodings)
    if average_embedding is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate face embedding.")

    # Save user to database
    user_data = {
        "id_number": id_number,
        "name": name,
        "role": role,
        "face_embedding": average_embedding.tolist(), # Store as list for MongoDB
        "on_site": False, # Default to off-site
        "registration_timestamp": datetime.utcnow(),
        "aadhar_path": str(aadhar_path) if aadhar_path else None,
        "role_id_path": str(role_id_path) if role_id_path else None,
        "face_photo_paths": face_photo_paths
    }

    if save_user(user_data):
        await load_known_faces() # Reload faces into memory after new registration
        logger.info(f"User {id_number} registered successfully.")
        return JSONResponse(content={"message": "User registered successfully", "user_id": id_number}, status_code=201)
    else:
        logger.error(f"Failed to save user {id_number} to database.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to register user.")

@app.get("/users/{user_id}")
async def get_user(user_id: str):
    """Retrieve details of a specific user."""
    try:
        user_data = await get_user_by_id(user_id)
        if user_data:
            # Prepare data for response, converting ObjectId and numpy array
            user_data['_id'] = str(user_data['_id'])
            if 'face_embedding' in user_data and isinstance(user_data['face_embedding'], np.ndarray):
                user_data['face_embedding'] = user_data['face_embedding'].tolist() # Convert back to list for JSON
            return user_data
        else:
            raise HTTPException(status_code=404, detail=f"User with ID '{user_id}' not found")
    except Exception as e:
        logger.error(f"Error fetching user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error fetching user: {str(e)}")

@app.get("/users")
async def list_users():
    """List all registered users."""
    try:
        users_data = await get_all_users()
        users_list = []
        for user in users_data:
            users_list.append({
                "id_number": user['id_number'],
                "name": user['name'],
                "role": user['role'],
                "on_site": user['on_site'],
                "aadhar_path": user.get('aadhar_path'),
                "role_id_path": user.get('role_id_path')
            })
        return {"users": users_list, "total": len(users_list)}
    except Exception as e:
        logger.error(f"Error listing all users: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error listing users: {str(e)}")

# --- Endpoints: Face Recognition & Access Control (from check_server/recognition.py) ---

@app.post("/recognize_face")
async def recognize_face(image: UploadFile = File(...), kiosk_type: str = Form(...)):
    """
    Performs face recognition on an uploaded image for entry/exit kiosks.
    `kiosk_type` can be "entry" or "exit".
    """
    if kiosk_type not in ["entry", "exit"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid kiosk_type. Must be 'entry' or 'exit'.")

    logger.info(f"Received face recognition request for {kiosk_type} kiosk.")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No image data received.")

    # Validate image quality before processing
    if not validate_image_quality(image_bytes):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Image quality is insufficient for recognition.")

    face_encodings = extract_face_encodings_from_bytes(image_bytes)

    if not face_encodings:
        logger.warning("No face detected in the provided image.")
        return JSONResponse(content={"status": "no_face", "message": "No face detected."}, status_code=200)

    # Assuming only one face for kiosk recognition
    face_encoding = face_encodings[0]

    if not known_face_encodings:
        logger.warning("No known faces loaded for recognition.")
        return JSONResponse(content={"status": "no_known_faces", "message": "No known faces registered."}, status_code=200)

    match, match_index, distance = compare_faces(known_face_encodings, face_encoding, EMBEDDING_THRESHOLD)
    confidence = calculate_confidence(distance, EMBEDDING_THRESHOLD)

    user_id = "Unknown"
    user_name = "Unknown"
    user_role = "Unknown"
    box_color = UNKNOWN_COLOR # Default to orange for unknown

    if match:
        user_id = known_face_ids[match_index]
        user_data = await get_user_by_id(user_id)

        if user_data:
            user_name = user_data['name']
            user_role = user_data['role']
            box_color = ROLE_COLORS.get(user_role, UNKNOWN_COLOR) # Use parsed ROLE_COLORS
            current_on_site_status = user_data['on_site']

            # Cooldown check
            if user_id in last_recognition_time:
                time_since_last_recognition = (datetime.utcnow() - last_recognition_time[user_id]).total_seconds()
                if time_since_last_recognition < COOLDOWN_PERIOD_SECONDS:
                    logger.info(f"User {user_name} ({user_id}) recognized but still in cooldown period.")
                    return JSONResponse(
                        content={"status": "cooldown", "message": f"Welcome back, {user_name}! Please wait a moment.", "user_id": user_id, "name": user_name, "confidence": confidence},
                        status_code=200
                    )

            if kiosk_type == "entry":
                if current_on_site_status:
                    message = f"Welcome back, {user_name}! You are already marked as on-site."
                    status_code = 200
                else:
                    await update_user_status(user_id, True)
                    await log_access_event(user_id, user_name, 'entry', 'face_recognition', confidence)
                    message = f"Entry granted for {user_name}. Welcome!"
                    status_code = 200
            elif kiosk_type == "exit":
                if not current_on_site_status:
                    message = f"Goodbye, {user_name}! You are already marked as off-site."
                    status_code = 200
                else:
                    await update_user_status(user_id, False)
                    await log_access_event(user_id, user_name, 'exit', 'face_recognition', confidence)
                    message = f"Exit granted for {user_name}. Goodbye!"
                    status_code = 200

            last_recognition_time[user_id] = datetime.utcnow() # Update last recognition time

            return JSONResponse(
                content={"status": "recognized", "message": message, "user_id": user_id, "name": user_name, "confidence": confidence},
                status_code=status_code
            )
        else:
            logger.error(f"Recognized face (ID: {user_id}) but user data not found in DB.")
            return JSONResponse(content={"status": "error", "message": "Recognized user data not found."}, status_code=500)
    else:
        logger.info(f"Unknown face detected. Distance: {distance:.2f}")
        return JSONResponse(content={"status": "unknown", "message": "Unknown person detected."}, status_code=200)


@app.get("/logs")
async def get_all_logs(user_id: Optional[str] = None, action: Optional[str] = None, limit: int = 100):
    """Retrieve access logs, optionally filtered by user_id or action."""
    logs = await get_access_logs(user_id=user_id, action=action, limit=limit)
    # Convert datetime objects to ISO format strings for JSON serialization
    for log in logs:
        if 'timestamp' in log and isinstance(log['timestamp'], datetime):
            log['timestamp'] = log['timestamp'].isoformat()
    return logs

@app.get("/users/{user_id}/status")
async def get_user_on_site_status(user_id: str):
    """Get a user's current on-site status."""
    user_data = await get_user_by_id(user_id)
    if user_data:
        return {"user_id": user_id, "on_site": user_data['on_site']}
    else:
        raise HTTPException(status_code=404, detail="User not found")

@app.post("/users/{user_id}/manual_entry")
async def manual_entry(user_id: str):
    """Manually log a user entry."""
    user_data = await get_user_by_id(user_id)
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")

    if await update_user_status(user_id, True):
        await log_access_event(user_id, user_data['name'], 'entry', 'manual')
        return {"status": "success", "message": f"Manual entry logged for {user_data['name']}."}
    else:
        raise HTTPException(status_code=500, detail="Failed to update user status for manual entry.")

@app.post("/users/{user_id}/manual_exit")
async def manual_exit(user_id: str):
    """Manually log a user exit."""
    user_data = await get_user_by_id(user_id)
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")

    if await update_user_status(user_id, False):
        await log_access_event(user_id, user_name=user_data['name'], action='exit', method='manual')
        return {"status": "success", "message": f"Manual exit logged for {user_data['name']}."}
    else:
        raise HTTPException(status_code=500, detail="Failed to update user status for manual exit.")

@app.post("/reload-faces")
async def reload_faces_endpoint():
    """Reloads all known face encodings from the database. Useful after new registrations."""
    await load_known_faces()
    return {"status": "success", "message": "Known faces reloaded successfully."}

# --- Endpoints: Scanner Control (from check_server/recognition.py) ---

@app.get("/scanner_status")
async def get_scanner_status():
    """Get the current scanning status of entry and exit kiosks."""
    return scanner_status

@app.post("/start_entry_scanning")
async def start_entry_scanning():
    """Starts the entry kiosk scanning."""
    scanner_status["entry_scanning"] = True
    logger.info("Entry scanning started.")
    return {"status": "success", "message": "Entry scanning started."}

@app.post("/stop_entry_scanning")
async def stop_entry_scanning():
    """Stops the entry kiosk scanning."""
    scanner_status["entry_scanning"] = False
    logger.info("Entry scanning stopped.")
    return {"status": "success", "message": "Entry scanning stopped."}

@app.post("/start_exit_scanning")
async def start_exit_scanning():
    """Starts the exit kiosk scanning."""
    scanner_status["exit_scanning"] = True
    logger.info("Exit scanning started.")
    return {"status": "success", "message": "Exit scanning started."}

@app.post("/stop_exit_scanning")
async def stop_exit_scanning():
    """Stops the exit kiosk scanning."""
    scanner_status["exit_scanning"] = False
    logger.info("Exit scanning stopped.")
    return {"status": "success", "message": "Exit scanning stopped."}

# --- Endpoints: Dashboard Data (from backend/routes/dashboardRoutes.js) ---

@app.get("/dashboard/activities")
async def dashboard_activities():
    """Get all activities (recent logs) for the dashboard."""
    return await get_dashboard_activities()

@app.get("/dashboard/personnel-breakdown")
async def personnel_breakdown():
    """Get personnel breakdown by role for currently on-site users."""
    return await get_personnel_breakdown()

@app.get("/dashboard/total-on-site")
async def total_on_site():
    """Get total personnel currently on site."""
    return {"totalOnSite": await get_total_on_site()}

@app.get("/dashboard/today-stats")
async def today_stats():
    """Get today's activity statistics."""
    return await get_today_stats()

# --- WebSocket for Video Feeds (from check_server/recognition.py) ---
# Note: For actual video streaming, you'd typically use a dedicated streaming protocol
# or send MJPEG over HTTP. WebSockets can work but might be less efficient for raw video.
# This implementation assumes sending processed frames.

def get_camera_stream(camera_index: int):
    """Initializes and returns a cv2.VideoCapture object for the given camera index."""
    if camera_index not in camera_streams or not camera_streams[camera_index].isOpened():
        cap = None
        for attempt in range(CAMERA_RETRY_ATTEMPTS):
            try:
                cap = cv2.VideoCapture(camera_index)
                if cap.isOpened():
                    logger.info(f"Successfully opened camera {camera_index} on attempt {attempt + 1}")
                    camera_streams[camera_index] = cap
                    return cap
                else:
                    logger.warning(f"Failed to open camera {camera_index} on attempt {attempt + 1}. Retrying in {CAMERA_RETRY_DELAY} seconds...")
                    time.sleep(CAMERA_RETRY_DELAY)
            except Exception as e:
                logger.error(f"Error opening camera {camera_index} on attempt {attempt + 1}: {e}")
                time.sleep(CAMERA_RETRY_DELAY)
        logger.error(f"Failed to open camera {camera_index} after {CAMERA_RETRY_ATTEMPTS} attempts.")
        return None
    return camera_streams[camera_index]

async def generate_frames(camera_index: int, kiosk_type: str):
    """Generates JPEG frames from the camera for WebSocket streaming."""
    cap = get_camera_stream(camera_index)
    if not cap:
        logger.error(f"Cannot open camera {camera_index} for {kiosk_type} feed.")
        return

    last_face_processing_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Failed to read frame from camera {camera_index}. Attempting to re-open...")
            cap = get_camera_stream(camera_index) # Try to re-open
            if not cap:
                break # Give up if re-opening fails
            continue

        # Only process faces if scanning is enabled for this kiosk type
        if (kiosk_type == "entry" and scanner_status["entry_scanning"]) or \
           (kiosk_type == "exit" and scanner_status["exit_scanning"]):
            current_time = time.time()
            if current_time - last_face_processing_time >= FACE_PROCESSING_INTERVAL:
                # Resize frame for faster face detection
                small_frame = cv2.resize(frame, (0, 0), fx=FACE_DETECTION_SCALE, fy=FACE_DETECTION_SCALE)
                rgb_small_frame = small_frame[:, :, ::-1] # Convert BGR to RGB

                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings_current_frame = face_recognition.face_encodings(rgb_small_frame, face_locations)

                for (top, right, bottom, left), face_encoding_current_frame in zip(face_locations, face_encodings_current_frame):
                    # Scale back up face locations
                    top *= int(1 / FACE_DETECTION_SCALE)
                    right *= int(1 / FACE_DETECTION_SCALE)
                    bottom *= int(1 / FACE_DETECTION_SCALE)
                    left *= int(1 / FACE_DETECTION_SCALE)

                    match, match_index, distance = compare_faces(known_face_encodings, face_encoding_current_frame, EMBEDDING_THRESHOLD)
                    confidence = calculate_confidence(distance, EMBEDDING_THRESHOLD)

                    user_id = "Unknown"
                    user_name = "Unknown"
                    user_role = "Unknown"
                    box_color = UNKNOWN_COLOR # Default to orange for unknown

                    if match:
                        user_id = known_face_ids[match_index]
                        user_data = await get_user_by_id(user_id) # Await the async function
                        if user_data:
                            user_name = user_data['name']
                            user_role = user_data['role']
                            box_color = ROLE_COLORS.get(user_role, UNKNOWN_COLOR)

                            # Cooldown check for logging
                            can_log = True
                            if user_id in last_recognition_time:
                                time_since_last_recognition = (datetime.utcnow() - last_recognition_time[user_id]).total_seconds()
                                if time_since_last_recognition < COOLDOWN_PERIOD_SECONDS:
                                    can_log = False

                            if can_log:
                                current_on_site_status = user_data['on_site']
                                if kiosk_type == "entry" and not current_on_site_status:
                                    await update_user_status(user_id, True)
                                    await log_access_event(user_id, user_name, 'entry', 'face_recognition', confidence)
                                    logger.info(f"Auto-entry for {user_name} ({user_id})")
                                elif kiosk_type == "exit" and current_on_site_status:
                                    await update_user_status(user_id, False)
                                    await log_access_event(user_id, user_name, 'exit', 'face_recognition', confidence)
                                    logger.info(f"Auto-exit for {user_name} ({user_id})")
                                last_recognition_time[user_id] = datetime.utcnow()

                    label = f"{user_name} ({user_role}) - {confidence:.1f}%"
                    frame = draw_face_bounding_box(frame, (top, right, bottom, left), label, box_color)
                last_face_processing_time = current_time

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            logger.error(f"Failed to encode frame from camera {camera_index}.")
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        await asyncio.sleep(1 / STREAM_FPS) # Control stream FPS

@app.get("/entry_video_feed")
async def entry_video_feed():
    """Video feed for the entry kiosk."""
    return StreamingResponse(generate_frames(ENTRY_CAMERA_INDEX, "entry"), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/exit_video_feed")
async def exit_video_feed():
    """Video feed for the exit kiosk."""
    return StreamingResponse(generate_frames(EXIT_CAMERA_INDEX, "exit"), media_type="multipart/x-mixed-replace; boundary=frame")


# To run the FastAPI application:
# Save this file as main.py
# Install dependencies: pip install fastapi uvicorn python-dotenv pymongo numpy opencv-python face_recognition
# Run from your terminal: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
