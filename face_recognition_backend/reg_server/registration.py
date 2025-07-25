# reg_server/registration.py
"""
FastAPI User Registration Server for Campus Face Recognition System
Handles user registration with face embedding generation through HTTP endpoints.
"""

import os
import json
import shutil
import numpy as np
import face_recognition
from pathlib import Path
from typing import List, Optional
import base64
import logging

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Import database and config utilities
from shared.database import initialize_database, save_user, check_user_exists, get_user_by_id, get_all_users
from shared.config import CARDS_DIR, FACES_DIR, VALID_ROLES, MAX_FILE_SIZE, LOG_LEVEL

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper()),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Campus Face Registration API",
    description="API for registering users in the campus face recognition system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class UserRegistrationResponse(BaseModel):
    success: bool
    message: str
    user_id: Optional[str] = None

class WebcamCaptureRequest(BaseModel):
    user_id: str
    image_data: str  # Base64 encoded image

# Utility functions
def create_directories():
    """Create necessary directories if they don't exist."""
    try:
        Path(CARDS_DIR).mkdir(exist_ok=True, parents=True) # parents=True to create parent dirs if needed
        Path(FACES_DIR).mkdir(exist_ok=True, parents=True)
        logger.info(f"Directories '{CARDS_DIR}' and '{FACES_DIR}' ensured.")
        return True
    except Exception as e:
        logger.critical(f"Error creating directories: {e}")
        return False

def validate_file_type(filename: str) -> bool:
    """Validate if the file has a valid image extension."""
    valid_extensions = ['.jpg', '.jpeg', '.png']
    file_ext = Path(filename).suffix.lower()
    return file_ext in valid_extensions

def save_uploaded_file(file: UploadFile, directory: str, prefix: str = "") -> str:
    """Save uploaded file to specified directory."""
    if not validate_file_type(file.filename):
        logger.warning(f"Invalid file type for {file.filename}.")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Supported: .jpg, .jpeg, .png"
        )
    
    # Ensure filename is safe and unique
    original_filename = Path(file.filename).stem
    file_ext = Path(file.filename).suffix
    unique_filename = f"{prefix}_{original_filename}{file_ext}" if prefix else f"{original_filename}{file_ext}"
    
    file_path = os.path.join(directory, unique_filename)
    
    try:
        # Read file content in chunks to handle large files efficiently
        with open(file_path, "wb") as buffer:
            while True:
                chunk = file.file.read(4096) # Read 4KB chunks
                if not chunk:
                    break
                buffer.write(chunk)
        logger.info(f"File saved to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error saving file {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

def extract_face_encoding(image_path: str):
    """Extract face encoding from an image file."""
    try:
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        
        if len(encodings) == 0:
            logger.warning(f"No face found in {image_path}.")
            return None
        elif len(encodings) > 1:
            logger.warning(f"Multiple faces found in {image_path}. Using the first one.")
        
        return encodings[0]
        
    except Exception as e:
        logger.error(f"Error processing image {image_path} for face encoding: {e}")
        return None

def generate_average_embedding(face_photo_paths: List[str]):
    """Generate average face embedding from multiple photos."""
    embeddings = []
    
    for photo_path in face_photo_paths:
        encoding = extract_face_encoding(photo_path)
        if encoding is not None:
            embeddings.append(encoding)
    
    if len(embeddings) == 0:
        return None
    
    return np.mean(embeddings, axis=0)

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize system on startup."""
    logger.info("Registration API starting up...")
    if not create_directories():
        logger.critical("Failed to create necessary directories. Exiting.")
        raise Exception("Failed to create directories")
    
    if not initialize_database():
        logger.critical("Failed to initialize database. Exiting.")
        raise Exception("Failed to initialize database")
    logger.info("Registration API startup complete.")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Campus Face Recognition Registration API",
        "version": "1.0.0",
        "endpoints": {
            "register": "/register",
            "webcam_capture": "/webcam-capture",
            "health": "/health",
            "users": "/users/{user_id}",
            "list_users": "/users"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "API is running"}

@app.post("/webcam-capture", response_model=UserRegistrationResponse)
async def capture_webcam_photo(request: WebcamCaptureRequest):
    """Save a base64 encoded webcam photo."""
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image_data)
        
        # Save to temporary file first (or directly to FACES_DIR)
        # Using a fixed name for simplicity, consider more robust naming for multiple captures
        filename = f"{request.user_id}_webcam_capture.jpg" # Changed for clarity and uniqueness
        file_path = os.path.join(FACES_DIR, filename)
        
        with open(file_path, "wb") as f:
            f.write(image_data)
        
        logger.info(f"Webcam photo saved for user {request.user_id} at {file_path}")
        return UserRegistrationResponse(
            success=True,
            message="Webcam photo saved successfully",
            user_id=request.user_id
        )
        
    except Exception as e:
        logger.error(f"Error saving webcam photo for user {request.user_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error saving webcam photo: {str(e)}"
        )

@app.post("/register", response_model=UserRegistrationResponse)
async def register_user(
    name: str = Form(..., description="Full name of the user"),
    role: str = Form(..., description="User role (student, professor, guard, maintenance)"),
    id_number: str = Form(..., description="Role-specific ID number"),
    aadhar_card: UploadFile = File(..., description="Aadhar card image"),
    role_id_card: UploadFile = File(..., description="Role-specific ID card image"),
    face_photo_1: UploadFile = File(..., description="First face photo"),
    face_photo_2: UploadFile = File(..., description="Second face photo"),
    webcam_photo: Optional[UploadFile] = File(None, description="Optional webcam photo")
):
    """Register a new user with uploaded files."""
    
    # Validate role
    if role not in VALID_ROLES:
        logger.warning(f"Invalid role '{role}' provided for user {id_number}.")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid role. Valid roles: {', '.join(VALID_ROLES)}"
        )
    
    # Validate file sizes
    files_to_check = [aadhar_card, role_id_card, face_photo_1, face_photo_2]
    if webcam_photo:
        files_to_check.append(webcam_photo)
    
    for file in files_to_check:
        if file.size > MAX_FILE_SIZE:
            logger.warning(f"File {file.filename} exceeds maximum size.")
            raise HTTPException(
                status_code=400,
                detail=f"File {file.filename} exceeds maximum size of {MAX_FILE_SIZE/1024/1024}MB"
            )
    
    # Check for duplicate user
    if check_user_exists(id_number): # Using updated check_user_exists
        logger.warning(f"Attempted to register duplicate user ID: {id_number}.")
        raise HTTPException(
            status_code=400,
            detail=f"User with ID '{id_number}' already exists."
        )
    
    aadhar_path = None
    role_id_path = None
    face_photo_paths = []
    
    try:
        # Save card documents
        aadhar_path = save_uploaded_file(aadhar_card, CARDS_DIR, f"{id_number}_aadhar")
        role_id_path = save_uploaded_file(role_id_card, CARDS_DIR, f"{id_number}_role_id")
        
        # Save face photos
        face_path_1 = save_uploaded_file(face_photo_1, FACES_DIR, f"{id_number}_face_1")
        face_path_2 = save_uploaded_file(face_photo_2, FACES_DIR, f"{id_number}_face_2")
        face_photo_paths.extend([face_path_1, face_path_2])
        
        # Add webcam photo if provided
        if webcam_photo:
            webcam_path = save_uploaded_file(webcam_photo, FACES_DIR, f"{id_number}_webcam")
            face_photo_paths.append(webcam_path)
        
        # Generate face embeddings
        average_embedding = generate_average_embedding(face_photo_paths)
        if average_embedding is None:
            logger.warning(f"No valid face encodings found for user {id_number}.")
            # Clean up uploaded files on failure
            for path in [aadhar_path, role_id_path] + face_photo_paths:
                if path and os.path.exists(path):
                    os.remove(path)
            
            raise HTTPException(
                status_code=400,
                detail="No valid face encodings found in uploaded photos. Please ensure clear face visibility."
            )
        
        # Save to database using the updated save_user function
        if save_user(role, name, id_number, average_embedding.tolist(), aadhar_path, role_id_path, face_photo_paths):
            logger.info(f"User '{name}' (ID: {id_number}) successfully registered.")
            return UserRegistrationResponse(
                success=True,
                message=f"User '{name}' successfully registered as {role.title()}",
                user_id=id_number
            )
        else:
            logger.error(f"Failed to save user '{name}' (ID: {id_number}) to database after processing.")
            # Clean up uploaded files on failure
            for path in [aadhar_path, role_id_path] + face_photo_paths:
                if path and os.path.exists(path):
                    os.remove(path)
            
            raise HTTPException(
                status_code=500,
                detail="Failed to save user to database."
            )
    
    except HTTPException:
        raise # Re-raise FastAPI HTTPExceptions directly
    except Exception as e:
        logger.critical(f"Unexpected error during registration for user {id_number}: {str(e)}", exc_info=True)
        # Attempt cleanup in case of unexpected errors too
        for path in [aadhar_path, role_id_path] + face_photo_paths:
            if path and os.path.exists(path):
                os.remove(path)
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during registration: {str(e)}. Files cleaned up."
        )

@app.get("/users/{user_id}")
async def get_user(user_id: str):
    """Get user information by ID."""
    try:
        user_data = get_user_by_id(user_id)
        
        if user_data:
            # Return a subset of data for security/privacy if needed, or all.
            # Example: Don't expose file paths or embeddings directly if not intended for client.
            return {
                "id_number": user_data['id_number'],
                "name": user_data['name'],
                "role": user_data['role'],
                "on_site": user_data['on_site'],
                "aadhar_path": user_data['aadhar_path'], # Now directly available
                "role_id_path": user_data['role_id_path'] # Now directly available
            }
        else:
            raise HTTPException(status_code=404, detail=f"User with ID '{user_id}' not found")
            
    except Exception as e:
        logger.error(f"Error fetching user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error fetching user: {str(e)}")

@app.get("/users")
async def list_users():
    """List all registered users."""
    try:
        users_data = get_all_users() # Use the shared database utility
        
        # Prepare a list of dictionaries with desired fields
        users_list = []
        for user in users_data:
            users_list.append({
                "id_number": user['id_number'],
                "name": user['name'],
                "role": user['role'],
                "on_site": user['on_site'],
                "aadhar_path": user['aadhar_path'], # Include if desired for display
                "role_id_path": user['role_id_path'] # Include if desired for display
            })
        
        return {"users": users_list, "total": len(users_list)}
            
    except Exception as e:
        logger.error(f"Error listing all users: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error listing users: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level=LOG_LEVEL.lower())