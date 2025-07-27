# shared/validation.py

"""
Input validation utilities for the Campus Face Recognition System.
Handles validation of file uploads, user data, and image quality.
"""

import os
import re
import cv2
import imghdr # To check image type by content
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Union

from fastapi import UploadFile, HTTPException, status

# Configuration values (previously from config.py, now defined directly)
# These should be consistent with your .env and main.py
VALID_EXTENSIONS = ['.jpeg', '.jpg', '.png', '.pdf'] # Assuming these are the relevant extensions
MAX_FILE_SIZE = 10485760 # 10MB
VALID_ROLES = ['student', 'professor', 'guard', 'maintenance']
ALLOWED_MIME_TYPES = ['image/jpeg', 'image/jpg', 'image/png', 'application/pdf']

logger = logging.getLogger(__name__)

def validate_file_upload(file: UploadFile, allowed_types: Optional[List[str]] = None,
                             max_size: int = MAX_FILE_SIZE) -> bool:
    """
    Validates an uploaded file for type, size, and content.

    Args:
        file (UploadFile): The uploaded file object from FastAPI.
        allowed_types (Optional[List[str]]): List of allowed file extensions (e.g., ['.jpg', '.png']).
                                             If None, uses global ALLOWED_MIME_TYPES.
        max_size (int): Maximum allowed file size in bytes.

    Returns:
        bool: True if validation passes, raises HTTPException otherwise.
    """
    if allowed_types is None:
        allowed_types = ALLOWED_MIME_TYPES # Use ALLOWED_MIME_TYPES for content type check

    # 1. Check file size
    file.file.seek(0, os.SEEK_END)
    file_size = file.file.tell()
    file.file.seek(0) # Reset file pointer to the beginning

    if file_size == 0:
        logger.warning(f"Uploaded file '{file.filename}' is empty.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file cannot be empty.")
    
    if file_size > max_size:
        logger.warning(f"Uploaded file '{file.filename}' size ({file_size} bytes) exceeds max allowed ({max_size} bytes).")
        raise HTTPException(status_code=status.HTTP_413_PAYLOAD_TOO_LARGE, detail=f"File too large. Max size is {max_size / (1024 * 1024):.2f} MB.")

    # 2. Check file extension (basic check, can be spoofed)
    file_extension = Path(file.filename).suffix.lower()
    # Ensure VALID_EXTENSIONS is used here, not ALLOWED_MIME_TYPES
    if file_extension not in VALID_EXTENSIONS: 
        logger.warning(f"Uploaded file '{file.filename}' has disallowed extension: {file_extension}.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid file type. Allowed extensions are: {', '.join(VALID_EXTENSIONS)}")

    # 3. Check MIME type (more reliable than extension, but still can be spoofed)
    if file.content_type not in allowed_types:
        logger.warning(f"Uploaded file '{file.filename}' has disallowed MIME type: {file.content_type}.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid content type. Allowed types are: {', '.join(allowed_types)}")

    # 4. Read a small chunk to guess the actual image type for image files
    # This helps prevent non-image files from being processed as images
    if file.content_type.startswith('image/'):
        contents = file.file.read(512) # Read first 512 bytes
        file.file.seek(0) # Reset file pointer
        image_type = imghdr.what(None, h=contents)
        if image_type not in ['jpeg', 'png', 'gif', 'tiff', 'bmp']: # Common image types
            logger.warning(f"Uploaded image file '{file.filename}' has unexpected content type based on header: {image_type}.")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File content does not appear to be a valid image.")

    logger.info(f"File '{file.filename}' validated successfully (size: {file_size} bytes, type: {file.content_type}).")
    return True

def validate_image_quality(image_bytes: bytes, min_width: int = 200, min_height: int = 200, min_face_size: int = 80) -> bool:
    """
    Performs basic image quality checks for face recognition suitability.
    - Checks minimum dimensions.
    - (Optional: Could add checks for blur, brightness, multiple faces here).
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            logger.warning("Could not decode image bytes for quality validation.")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid image data provided.")

        height, width, _ = img.shape

        if width < min_width or height < min_height:
            logger.warning(f"Image dimensions too small: {width}x{height}. Minimum: {min_width}x{min_height} pixels.")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image dimensions too small. Minimum: {min_width}x{min_height} pixels."
            )

        # Additional checks could include:
        # - Brightness/contrast: Calculate mean pixel value, standard deviation.
        # - Blur detection: Using Laplacian variance.
        # - Face clarity (requires a face detector, could be integrated with face_recognition)

        logger.info(f"Image quality validated (dimensions: {width}x{height}).")
        return True
    except HTTPException:
        raise # Re-raise FastAPI HTTPExceptions
    except Exception as e:
        logger.error(f"Error validating image quality: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error validating image quality: {str(e)}")


def validate_user_data(name: str, role: str, id_number: str) -> bool:
    """
    Performs comprehensive validation for core user registration data.

    Args:
        name (str): Full name of the user.
        role (str): Role of the user.
        id_number (str): ID number of the user.

    Returns:
        bool: True if all user data is valid, raises HTTPException otherwise.
    """
    if not name or not name.strip():
        logger.warning("User name is empty or whitespace.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Name cannot be empty.")
    if len(name) > 100: # Example length limit
        logger.warning(f"User name too long: {len(name)} characters.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Name is too long.")

    validate_user_role(role) # Will raise HTTPException if invalid
    validate_id_number(id_number) # Will raise HTTPException if invalid

    logger.info(f"User data validated for {name} ({id_number}).")
    return True

def validate_user_role(role: str) -> bool:
    """
    Validates if the provided role is one of the allowed roles.
    """
    if not role or role not in VALID_ROLES:
        logger.warning(f"Invalid user role: '{role}'. Allowed roles: {VALID_ROLES}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid role. Must be one of: {', '.join(VALID_ROLES)}")
    return True

def validate_id_number(id_number: str) -> bool:
    """
    Validates the format and content of the ID number.
    Example: Alphanumeric, specific length, or regex pattern.
    """
    if not id_number or not id_number.strip():
        logger.warning("ID number is empty or whitespace.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="ID number cannot be empty.")
    
    # Example: Check for alphanumeric and length between 5 and 20
    if not re.fullmatch(r"^[a-zA-Z0-9]{5,20}$", id_number):
        logger.warning(f"Invalid ID number format: '{id_number}'.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="ID number must be alphanumeric and between 5-20 characters.")
    
    return True
