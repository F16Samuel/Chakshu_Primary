"""
Face Recognition System - FastAPI Backend
Comprehensive system with MongoDB integration and all required APIs
"""

import os
import json
import base64
import cv2
import numpy as np
import face_recognition
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pathlib import Path
import shutil
import uuid

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient
import gridfs
from bson import ObjectId
from dotenv import load_dotenv
import asyncio
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "campus_access")
CARDS_DIR = Path(os.getenv("CARDS_DIR", "./uploads/cards"))
FACES_DIR = Path(os.getenv("FACES_DIR", "./uploads/faces"))

# Create directories
CARDS_DIR.mkdir(parents=True, exist_ok=True)
FACES_DIR.mkdir(parents=True, exist_ok=True)

# Role configurations
VALID_ROLES = json.loads(os.getenv("VALID_ROLES", '["student", "professor", "guard", "maintenance"]'))
ROLE_COLORS = json.loads(os.getenv("ROLE_COLORS", '{"student": [0, 255, 0], "professor": [255, 0, 0], "guard": [255, 0, 255], "maintenance": [0, 255, 255]}'))
UNKNOWN_COLOR = tuple(json.loads(os.getenv("UNKNOWN_COLOR", "[0, 165, 255]")))

# Initialize FastAPI app
app = FastAPI(
    title="Face Recognition System",
    description="Campus Access Control System with Face Recognition",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB client
client = None
db = None
fs = None

# Pydantic models
class UserStatus(BaseModel):
    status: str = Field(..., description="User status: 'entry' or 'exit'")

class PersonnelBreakdown(BaseModel):
    role: str
    count: int
    on_site: int

class ActivityRecord(BaseModel):
    id: str
    user_id: str
    name: str
    role: str
    action: str
    timestamp: datetime
    confidence: Optional[float] = None

class DashboardStats(BaseModel):
    total_entries: int
    total_exits: int
    failed_attempts: int
    unique_visitors: int

class RecognitionResult(BaseModel):
    recognized: bool
    user_id: Optional[str] = None
    name: Optional[str] = None
    role: Optional[str] = None
    confidence: Optional[float] = None
    action: Optional[str] = None
    timestamp: datetime

async def connect_to_mongo():
    """Connect to MongoDB"""
    global client, db, fs
    try:
        client = AsyncIOMotorClient(MONGO_URI)
        db = client[MONGO_DB_NAME]
        # Remove GridFS initialization for now - it's not used in the current code
        fs = None
        
        # Test connection
        await client.admin.command('ping')
        logger.info("Connected to MongoDB successfully")
        
        # Create indexes
        await create_indexes()
        
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise

async def create_indexes():
    """Create database indexes for better performance"""
    try:
        # Users collection indexes
        await db.users.create_index("id_number", unique=True)
        await db.users.create_index("name")
        await db.users.create_index("role")
        
        # Activities collection indexes
        await db.activities.create_index("timestamp")
        await db.activities.create_index("user_id")
        await db.activities.create_index([("timestamp", -1)])
        
        logger.info("Database indexes created successfully")
    except Exception as e:
        logger.error(f"Failed to create indexes: {e}")

async def close_mongo_connection():
    """Close MongoDB connection"""
    global client
    if client:
        client.close()

# Face recognition utilities
class FaceRecognitionManager:
    def __init__(self):
        self.known_encodings = []
        self.known_users = []
        self.load_encodings()
    
    def load_encodings(self):
        pass
    
    async def _async_load_encodings(self):
        """Async method to load encodings from database"""
        try:
            users = await db.users.find({"face_encodings": {"$exists": True}}).to_list(None)
            self.known_encodings = []
            self.known_users = []
            
            for user in users:
                if "face_encodings" in user and user["face_encodings"]:
                    for encoding_data in user["face_encodings"]:
                        encoding = np.array(encoding_data)
                        self.known_encodings.append(encoding)
                        self.known_users.append({
                            "id": str(user["_id"]),
                            "name": user["name"],
                            "role": user["role"],
                            "id_number": user["id_number"]
                        })
            
            logger.info(f"Loaded {len(self.known_encodings)} face encodings")
        except Exception as e:
            logger.error(f"Failed to load encodings: {e}")
    
    async def add_user_encodings(self, user_id: str, face_images: List[UploadFile]):
        """Process and store face encodings for a user"""
        encodings = []
        
        for image_file in face_images:
            try:
                # Read image data
                image_data = await image_file.read()
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Find face encodings
                face_locations = face_recognition.face_locations(rgb_image)
                face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
                
                if face_encodings:
                    encodings.extend([encoding.tolist() for encoding in face_encodings])
                else:
                    logger.warning(f"No face found in image {image_file.filename}")
                    
            except Exception as e:
                logger.error(f"Error processing image {image_file.filename}: {e}")
        
        if encodings:
            # Update user with encodings
            await db.users.update_one(
                {"_id": ObjectId(user_id)},
                {"$set": {"face_encodings": encodings}}
            )
            
            # Reload encodings
            await self._async_load_encodings()
            
        return len(encodings)
    
    async def recognize_face(self, image_data: bytes) -> Dict[str, Any]:
        """Recognize face from image data"""
        try:
            # Decode image
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Find faces in the image
            face_locations = face_recognition.face_locations(rgb_image)
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            if not face_encodings:
                return {
                    "recognized": False,
                    "error": "No face detected in the image"
                }
            
            # Compare with known faces
            for face_encoding in face_encodings:
                if self.known_encodings:
                    matches = face_recognition.compare_faces(self.known_encodings, face_encoding, tolerance=0.6)
                    face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
                    
                    if any(matches):
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            confidence = 1 - face_distances[best_match_index]
                            user = self.known_users[best_match_index]
                            
                            # Log activity
                            await self._log_activity(user, confidence)
                            
                            return {
                                "recognized": True,
                                "user_id": user["id"],
                                "name": user["name"],
                                "role": user["role"],
                                "confidence": float(confidence),
                                "timestamp": datetime.now(timezone.utc)
                            }
            
            # No match found
            await self._log_failed_attempt()
            return {
                "recognized": False,
                "confidence": 0.0,
                "timestamp": datetime.now(timezone.utc)
            }
            
        except Exception as e:
            logger.error(f"Face recognition error: {e}")
            return {
                "recognized": False,
                "error": str(e)
            }
    
    async def _log_activity(self, user: Dict, confidence: float):
        """Log successful recognition activity"""
        # Determine if it's entry or exit based on current status
        current_status = await db.users.find_one({"_id": ObjectId(user["id"])})
        action = "exit" if current_status and current_status.get("on_site", False) else "entry"
        
        # Update user status
        await db.users.update_one(
            {"_id": ObjectId(user["id"])},
            {
                "$set": {
                    "on_site": action == "entry",
                    "last_activity": datetime.now(timezone.utc)
                }
            }
        )
        
        # Log activity
        activity = {
            "user_id": user["id"],
            "name": user["name"],
            "role": user["role"],
            "action": action,
            "confidence": confidence,
            "timestamp": datetime.now(timezone.utc),
            "type": "recognition"
        }
        
        await db.activities.insert_one(activity)
    
    async def _log_failed_attempt(self):
        """Log failed recognition attempt"""
        activity = {
            "user_id": None,
            "name": "Unknown",
            "role": "unknown",
            "action": "failed_attempt",
            "confidence": 0.0,
            "timestamp": datetime.now(timezone.utc),
            "type": "failed_recognition"
        }
        
        await db.activities.insert_one(activity)

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    await connect_to_mongo()
    await face_manager._async_load_encodings()

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down FastAPI application...")
    await close_mongo_connection()
    logger.info("FastAPI application shut down.")

# API Routes

@app.get("/health")
async def health_check():
    """Check API health status"""
    try:
        # Test database connection
        await client.admin.command('ping')
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc),
            "database": "connected"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database connection failed: {str(e)}"
        )

@app.get("/api/system/status")
async def get_system_status():
    """Get system status - alias for health check"""
    try:
        # Test database connection
        await client.admin.command('ping')
        return {
            "status": "online",
            "database": "connected",
            "timestamp": datetime.now(timezone.utc),
            "face_encodings_loaded": len(face_manager.known_encodings)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"System unavailable: {str(e)}"
        )
    
@app.get("/users")
async def get_users():
    """Get list of all users"""
    try:
        users = await db.users.find({}, {
            "name": 1,
            "role": 1,
            "id_number": 1,
            "on_site": 1,
            "last_activity": 1
        }).to_list(None)
        
        # Convert ObjectId to string
        for user in users:
            user["id"] = str(user["_id"])
            del user["_id"]
        
        return {"users": users}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch users: {str(e)}"
        )

@app.put("/users/{user_id}")
async def update_user_status(user_id: str, status_update: UserStatus):
    """Manually update user entry/exit status"""
    try:
        if status_update.status not in ["entry", "exit"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Status must be 'entry' or 'exit'"
            )
        
        # Update user status
        result = await db.users.update_one(
            {"_id": ObjectId(user_id)},
            {
                "$set": {
                    "on_site": status_update.status == "entry",
                    "last_activity": datetime.now(timezone.utc)
                }
            }
        )
        
        if result.matched_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Get user details for activity log
        user = await db.users.find_one({"_id": ObjectId(user_id)})
        
        # Log manual activity
        activity = {
            "user_id": user_id,
            "name": user["name"],
            "role": user["role"],
            "action": status_update.status,
            "confidence": 1.0,
            "timestamp": datetime.now(timezone.utc),
            "type": "manual"
        }
        
        await db.activities.insert_one(activity)
        
        return {
            "message": f"User status updated to {status_update.status}",
            "user_id": user_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update user status: {str(e)}"
        )

@app.post("/register")
async def register_personnel(
    name: str = Form(...),
    role: str = Form(...),
    id_number: str = Form(...),
    aadhar_card: UploadFile = File(...),
    role_id_card: UploadFile = File(...),
    face_photos: List[UploadFile] = File(...)
):
    """Register new personnel with face recognition"""
    try:
        # Validate role
        if role.lower() not in VALID_ROLES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid role. Must be one of: {VALID_ROLES}"
            )
        
        # Check if user already exists
        existing_user = await db.users.find_one({"id_number": id_number})
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this ID number already exists"
            )
        
        # Validate face photos
        if len(face_photos) < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one face photo is required"
            )
        
        # Create user document
        user_doc = {
            "name": name,
            "role": role.lower(),
            "id_number": id_number,
            "on_site": False,
            "registered_at": datetime.now(timezone.utc),
            "last_activity": None
        }
        
        # Insert user
        result = await db.users.insert_one(user_doc)
        user_id = str(result.inserted_id)
        
        # Save card files
        user_cards_dir = CARDS_DIR / user_id
        user_cards_dir.mkdir(parents=True, exist_ok=True)
        
        # Save Aadhar card
        aadhar_path = user_cards_dir / f"aadhar_{aadhar_card.filename}"
        with open(aadhar_path, "wb") as f:
            shutil.copyfileobj(aadhar_card.file, f)
        
        # Save role ID card
        role_id_path = user_cards_dir / f"role_id_{role_id_card.filename}"
        with open(role_id_path, "wb") as f:
            shutil.copyfileobj(role_id_card.file, f)
        
        # Process face photos and create encodings
        encodings_count = await face_manager.add_user_encodings(user_id, face_photos)
        
        if encodings_count == 0:
            # Cleanup if no face encodings were created
            await db.users.delete_one({"_id": ObjectId(user_id)})
            shutil.rmtree(user_cards_dir, ignore_errors=True)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid face encodings could be created from the provided photos"
            )
        
        return {
            "message": "Personnel registered successfully",
            "user_id": user_id,
            "name": name,
            "role": role,
            "face_encodings_created": encodings_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )

@app.post("/recognize_face")
async def recognize_face(file: UploadFile = File(...)):
    """Perform facial recognition from uploaded image"""
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be an image"
            )
        
        # Read image data
        image_data = await file.read()
        
        # Perform recognition
        result = await face_manager.recognize_face(image_data)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face recognition error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Face recognition failed: {str(e)}"
        )

@app.get("/dashboard/activities")
async def get_recent_activities():
    """Get recent activities (entries, exits, attempts)"""
    try:
        activities = await db.activities.find({}).sort("timestamp", -1).limit(50).to_list(50)
        
        # Convert ObjectId to string and format response
        formatted_activities = []
        for activity in activities:
            formatted_activity = {
                "id": str(activity["_id"]),
                "user_id": activity.get("user_id"),
                "name": activity["name"],
                "role": activity["role"],
                "action": activity["action"],
                "timestamp": activity["timestamp"],
                "confidence": activity.get("confidence", 0.0)
            }
            formatted_activities.append(formatted_activity)
        
        return {"activities": formatted_activities}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch activities: {str(e)}"
        )

@app.get("/dashboard/personnel-breakdown")
async def get_personnel_breakdown():
    """Get breakdown of personnel on-site by roles"""
    try:
        pipeline = [
            {
                "$group": {
                    "_id": "$role",
                    "total": {"$sum": 1},
                    "on_site": {
                        "$sum": {
                            "$cond": [{"$eq": ["$on_site", True]}, 1, 0]
                        }
                    }
                }
            }
        ]
        
        results = await db.users.aggregate(pipeline).to_list(None)
        
        breakdown = []
        for result in results:
            breakdown.append({
                "role": result["_id"],
                "count": result["total"],
                "on_site": result["on_site"]
            })
        
        return {"breakdown": breakdown}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get personnel breakdown: {str(e)}"
        )

@app.get("/dashboard/total-on-site")
async def get_total_on_site():
    """Get total number of personnel currently on-site"""
    try:
        total = await db.users.count_documents({"on_site": True})
        return {"total_on_site": total}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get total on-site: {str(e)}"
        )

@app.get("/dashboard/today-stats")
async def get_today_stats():
    """Get today's statistics"""
    try:
        # Get today's date range
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = today_start.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        # Aggregate today's activities
        pipeline = [
            {
                "$match": {
                    "timestamp": {"$gte": today_start, "$lte": today_end}
                }
            },
            {
                "$group": {
                    "_id": "$action",
                    "count": {"$sum": 1}
                }
            }
        ]
        
        results = await db.activities.aggregate(pipeline).to_list(None)
        
        stats = {
            "total_entries": 0,
            "total_exits": 0,
            "failed_attempts": 0,
            "unique_visitors": 0
        }
        
        for result in results:
            action = result["_id"]
            count = result["count"]
            
            if action == "entry":
                stats["total_entries"] = count
            elif action == "exit":
                stats["total_exits"] = count
            elif action == "failed_attempt":
                stats["failed_attempts"] = count
        
        # Get unique visitors today
        unique_visitors = await db.activities.distinct(
            "user_id",
            {
                "timestamp": {"$gte": today_start, "$lte": today_end},
                "user_id": {"$ne": None}
            }
        )
        stats["unique_visitors"] = len(unique_visitors)
        
        return stats
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get today's stats: {str(e)}"
        )


face_manager = FaceRecognitionManager()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)