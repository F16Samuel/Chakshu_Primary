# shared/config.py
import os
from dotenv import load_dotenv
from pathlib import Path
import ast

# Determine the base directory of the 'face' project, assuming config.py is in 'shared'
# And 'shared' is a subdirectory of the main 'face' project directory.
# So, BASE_DIR will be the 'face' directory.
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from the base .env file (e.g., face/.env)
# This should be loaded first to establish defaults
load_dotenv(BASE_DIR / ".env")

# --- Database Configuration ---
DB_NAME = os.getenv("DB_NAME", "campus_access.db")
# If DB_PATH is relative, ensure it's relative to the project root, not shared/config.py
_db_path_env = os.getenv("DB_PATH", "./campus_access.db")
# Resolve DB_PATH based on whether it's absolute or relative to BASE_DIR
if Path(_db_path_env).is_absolute():
    DB_PATH = Path(_db_path_env)
else:
    DB_PATH = BASE_DIR / _db_path_env

# --- Directory Paths ---
# These paths are relative to BASE_DIR (the 'face' project root)
CARDS_DIR = BASE_DIR / os.getenv("CARDS_DIR", "./cards")
FACES_DIR = BASE_DIR / os.getenv("FACES_DIR", "./faces")

# --- Face Recognition Settings ---
EMBEDDING_THRESHOLD = float(os.getenv("EMBEDDING_THRESHOLD", "0.6"))
# RECOGNITION_FPS and STREAM_FPS are specific to check_server, but can have defaults here
RECOGNITION_FPS = float(os.getenv("RECOGNITION_FPS", "2.0"))
STREAM_FPS = float(os.getenv("STREAM_FPS", "60.0"))
FACE_DETECTION_SCALE = float(os.getenv("FACE_DETECTION_SCALE", "0.25"))
FACE_PROCESSING_INTERVAL = float(os.getenv("FACE_PROCESSING_INTERVAL", "0.5"))

# --- File Upload Settings ---
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10485760")) # Default 10MB
# Convert comma-separated string to list
VALID_EXTENSIONS = [ext.strip() for ext in os.getenv("VALID_EXTENSIONS", ".jpg,.jpeg,.png").split(',')]
ALLOWED_MIME_TYPES = [mt.strip() for mt in os.getenv("ALLOWED_MIME_TYPES", "image/jpeg,image/jpg,image/png").split(',')]

# --- Server Configuration (Defaults, overridden by service-specific .env) ---
REGISTRATION_HOST = os.getenv("REGISTRATION_HOST", "0.0.0.0")
REGISTRATION_PORT = int(os.getenv("REGISTRATION_PORT", "8000"))
RECOGNITION_HOST = os.getenv("RECOGNITION_HOST", "0.0.0.0")
RECOGNITION_PORT = int(os.getenv("RECOGNITION_PORT", "8001"))

# --- Camera Configuration ---
ENTRY_CAMERA_INDEX = int(os.getenv("ENTRY_CAMERA_INDEX", "0"))
EXIT_CAMERA_INDEX = int(os.getenv("EXIT_CAMERA_INDEX", "0"))
CAMERA_FALLBACK_ENABLED = os.getenv("CAMERA_FALLBACK_ENABLED", "false").lower() == 'true'
FRAME_TIMEOUT = int(os.getenv("FRAME_TIMEOUT", "30")) # Seconds before considering camera disconnected
CAMERA_RETRY_ATTEMPTS = int(os.getenv("CAMERA_RETRY_ATTEMPTS", "3"))
CAMERA_RETRY_DELAY = int(os.getenv("CAMERA_RETRY_DELAY", "2"))

# --- Security Settings ---
CORS_ORIGINS = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(',')]
CORS_CREDENTIALS = os.getenv("CORS_CREDENTIALS", "true").lower() == 'true'

# --- Logging ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper() # INFO, DEBUG, WARNING, ERROR, CRITICAL
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == 'true'

# --- Valid User Roles ---
VALID_ROLES = [role.strip() for role in os.getenv("VALID_ROLES", "student,professor,guard,maintenance").split(',')]

# --- Registration Specific Settings ---
REQUIRE_WEBCAM_PHOTO = os.getenv("REQUIRE_WEBCAM_PHOTO", "false").lower() == 'true'
MIN_FACE_PHOTOS = int(os.getenv("MIN_FACE_PHOTOS", "2"))
MAX_FACE_PHOTOS = int(os.getenv("MAX_FACE_PHOTOS", "5"))

# --- API Configuration (Defaults for generic use if not overridden) ---
API_TITLE = os.getenv("API_TITLE", "Campus Face Recognition System API")
API_DESCRIPTION = os.getenv("API_DESCRIPTION", "Central API for Campus Face Recognition System")
API_VERSION = os.getenv("API_VERSION", "1.0.0")

# --- Scanner Control (Default status) ---
DEFAULT_ENTRY_SCANNING = os.getenv("DEFAULT_ENTRY_SCANNING", "false").lower() == 'true'
DEFAULT_EXIT_SCANNING = os.getenv("DEFAULT_EXIT_SCANNING", "false").lower() == 'true'

# --- Color Configuration (BGR format) ---
UNKNOWN_COLOR = (int(os.getenv("UNKNOWN_COLOR_B", "0")), int(os.getenv("UNKNOWN_COLOR_G", "165")), int(os.getenv("UNKNOWN_COLOR_R", "255"))) # Orange
ROLE_COLORS = {
    "student": (int(os.getenv("STUDENT_COLOR_B", "0")), int(os.getenv("STUDENT_COLOR_G", "255")), int(os.getenv("STUDENT_COLOR_R", "0"))), # Green
    "professor": (int(os.getenv("PROFESSOR_COLOR_B", "255")), int(os.getenv("PROFESSOR_COLOR_G", "0")), int(os.getenv("PROFESSOR_COLOR_R", "0"))), # Blue
    "guard": (int(os.getenv("GUARD_COLOR_B", "255")), int(os.getenv("GUARD_COLOR_G", "0")), int(os.getenv("GUARD_COLOR_R", "255"))), # Magenta
    "maintenance": (int(os.getenv("MAINTENANCE_COLOR_B", "0")), int(os.getenv("MAINTENANCE_COLOR_G", "255")), int(os.getenv("MAINTENANCE_COLOR_R", "255"))) # Cyan
}

# Performance Tuning (from check/.env)
MAX_CONCURRENT_STREAMS = int(os.getenv("MAX_CONCURRENT_STREAMS", "2"))
MEMORY_CLEANUP_INTERVAL = int(os.getenv("MEMORY_CLEANUP_INTERVAL", "300"))

# Registration server specific
ENABLE_RATE_LIMITING = os.getenv("ENABLE_RATE_LIMITING", "true").lower() == 'true'
MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "60"))
COOLDOWN_PERIOD_SECONDS = int(os.getenv("COOLDOWN_PERIOD_SECONDS", "5"))