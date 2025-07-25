# shared/__init__.py
# This file simplifies imports from the 'shared' package

from .config import (
    DB_PATH, CARDS_DIR, FACES_DIR,
    EMBEDDING_THRESHOLD,
    REGISTRATION_HOST, REGISTRATION_PORT, RECOGNITION_HOST, RECOGNITION_PORT,
    CORS_ORIGINS, CORS_CREDENTIALS,
    LOG_LEVEL, API_TITLE, API_DESCRIPTION, API_VERSION,
    MAX_FILE_SIZE, VALID_EXTENSIONS, ALLOWED_MIME_TYPES,
    ENTRY_CAMERA_INDEX, EXIT_CAMERA_INDEX, CAMERA_FALLBACK_ENABLED,
    RECOGNITION_FPS, STREAM_FPS, FACE_DETECTION_SCALE, FACE_PROCESSING_INTERVAL,
    UNKNOWN_COLOR, ROLE_COLORS,
    REQUIRE_WEBCAM_PHOTO, MIN_FACE_PHOTOS, MAX_FACE_PHOTOS,
    DEFAULT_ENTRY_SCANNING, DEFAULT_EXIT_SCANNING,
    FRAME_TIMEOUT, CAMERA_RETRY_ATTEMPTS, CAMERA_RETRY_DELAY,
    MAX_CONCURRENT_STREAMS, MEMORY_CLEANUP_INTERVAL,
    ENABLE_RATE_LIMITING, MAX_REQUESTS_PER_MINUTE,
    VALID_ROLES # Added VALID_ROLES for validation module
)

from .database import (
    initialize_database,
    get_db_connection,
    save_user,
    check_user_exists,
    get_user_by_id,
    get_all_users,
    update_user_status,
    log_access_event,
    get_access_logs # Added get_access_logs
)

from .face_processing import (
    extract_face_encoding,
    extract_face_encodings_from_bytes,
    extract_face_locations_and_encodings_from_frame, # New function
    compare_faces,
    generate_average_embedding,
    calculate_confidence
)

from .validation import (
    validate_file_upload,
    validate_user_role,
    validate_id_format,
    sanitize_filename,
    validate_image_quality,
    validate_user_data
)