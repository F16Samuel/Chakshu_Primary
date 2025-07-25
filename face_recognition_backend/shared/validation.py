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

from .config import VALID_EXTENSIONS, MAX_FILE_SIZE, VALID_ROLES, ALLOWED_MIME_TYPES # Changed import

logger = logging.getLogger(__name__)

def validate_file_upload(file: UploadFile, allowed_types: Optional[List[str]] = None,
                             max_size: int = MAX_FILE_SIZE) -> bool:
    """
    Validates an uploaded file for type, size, and content.

    Args:
        file (UploadFile): The uploaded file object from FastAPI.
        allowed_types (Optional[List[str]]): List of allowed file extensions (e.g., ['.jpg', '.png']).
                                             If None, uses global VALID_EXTENSIONS.
        max_size (int): Maximum allowed file size in bytes.

    Returns:
        bool: True if validation passes, raises HTTPException otherwise.
    """
    if allowed_types is None:
        allowed_types = VALID_EXTENSIONS

    # 1. Check file size
    file.file.seek(0, os.SEEK_END)
    file_size = file.file.tell()
    file.file.seek(0) # Reset file pointer to the beginning

    if file_size == 0:
        logger.warning(f"Uploaded file {file.filename} is empty.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File {file.filename} is empty."
        )

    if file_size > max_size:
        logger.warning(f"File {file.filename} exceeds max size: {file_size} bytes (max {max_size} bytes).")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File {file.filename} exceeds maximum size of {max_size / (1024 * 1024):.2f}MB"
        )

    # 2. Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_types:
        logger.warning(f"Invalid file extension for {file.filename}: {file_ext}. Allowed: {allowed_types}.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type for {file.filename}. Supported: {', '.join(allowed_types)}"
        )

    # 3. Basic content type check (more robust checks might involve libraries like python-magic)
    # Read a small chunk to guess content type
    file_start_bytes = file.file.read(2048) # Read first 2KB
    file.file.seek(0) # Reset file pointer again

    guessed_type = imghdr.what(None, h=file_start_bytes)
    # Map imghdr types to common extensions
    type_map = {
        'jpeg': ['.jpg', '.jpeg'],
        'png': ['.png'],
        'gif': ['.gif'], # Not in our allowed_types, but good to know
        'tiff': ['.tif', '.tiff'],
        'bmp': ['.bmp']
    }
    # Check if the guessed type's common extensions are in our allowed list
    if guessed_type is None or not any(ext in allowed_types for ext in type_map.get(guessed_type, [])):
        logger.warning(f"File {file.filename} content type mismatch. Guessed: {guessed_type}, Expected: {file_ext}.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File content type mismatch for {file.filename}. Please upload a valid image."
        )

    logger.info(f"File {file.filename} validated successfully (size: {file_size}, type: {file_ext}).")
    return True

def validate_user_role(role: str) -> bool:
    """
    Validates if the provided role is one of the allowed roles.

    Args:
        role (str): The role string to validate.

    Returns:
        bool: True if the role is valid, raises HTTPException otherwise.
    """
    if role.lower() not in VALID_ROLES:
        logger.warning(f"Invalid user role: '{role}'. Allowed roles: {VALID_ROLES}.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role. Valid roles: {', '.join(VALID_ROLES)}"
        )
    return True

def validate_id_format(id_number: str, role: str) -> bool:
    """
    Validates the format of the ID number based on the user's role.
    This is a placeholder for more specific ID format validations.

    Args:
        id_number (str): The ID number to validate.
        role (str): The role associated with the ID number.

    Returns:
        bool: True if the ID format is considered valid, raises HTTPException otherwise.
    """
    # Example: Simple check for non-empty and alphanumeric (could be more complex)
    if not id_number or not re.match(r"^[a-zA-Z0-9_-]+$", id_number):
        logger.warning(f"Invalid ID format for '{id_number}' (Role: {role}). Must be alphanumeric/hyphens/underscores.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid ID number format. Must be alphanumeric with hyphens or underscores."
        )

    # Add role-specific checks if needed
    # if role.lower() == 'student' and not re.match(r"^S\d{7}$", id_number):
    #     raise HTTPException(status_code=400, detail="Student ID must start with 'S' followed by 7 digits.")
    # elif role.lower() == 'professor' and not re.match(r"^P\d{5}$", id_number):
    #     raise HTTPException(status_code=400, detail="Professor ID must start with 'P' followed by 5 digits.")

    logger.info(f"ID '{id_number}' for role '{role}' validated.")
    return True

def sanitize_filename(filename: str) -> str:
    """
    Sanitizes a filename to prevent directory traversal and other malicious uses.

    Args:
        filename (str): The original filename.

    Returns:
        str: The sanitized filename.
    """
    # Remove directory separators
    filename = filename.replace("\\", "_").replace("/", "_")
    # Remove any characters that are not alphanumeric, dashes, underscores, or dots
    filename = re.sub(r'[^\w\s.-]', '', filename).strip()
    # Replace spaces with underscores
    filename = re.sub(r'\s+', '_', filename)
    # Limit length
    filename = filename[:250]
    logger.debug(f"Filename sanitized: '{filename}'")
    return filename

def validate_image_quality(image_path_or_bytes: Union[str, bytes]) -> bool:
    """
    Performs basic image quality validation (e.g., minimum dimensions).
    This is a placeholder for more advanced quality checks.

    Args:
        image_path_or_bytes (Union[str, bytes]): Path to the image file or image bytes.

    Returns:
        bool: True if the image quality is deemed acceptable, raises HTTPException otherwise.
    """
    img = None
    if isinstance(image_path_or_bytes, str):
        if not os.path.exists(image_path_or_bytes):
            logger.error(f"Image file not found for quality validation: {image_path_or_bytes}")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Image file not found.")
        img = cv2.imread(image_path_or_bytes)
    elif isinstance(image_path_or_bytes, bytes):
        nparr = np.frombuffer(image_path_or_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        logger.error("Invalid input type for image quality validation. Must be path (str) or bytes.")
        raise TypeError("Input must be a file path (str) or image bytes.")

    if img is None:
        logger.warning("Could not load image for quality validation.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not load image for quality check. Invalid image data.")

    min_width, min_height = 100, 100 # Example minimum dimensions
    height, width, _ = img.shape

    if width < min_width or height < min_height:
        logger.warning(f"Image dimensions ({width}x{height}) are too small. Minimum: {min_width}x{min_height}.")
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
    validate_id_format(id_number, role) # Will raise HTTPException if invalid

    logger.info(f"User data for {name} (ID: {id_number}, Role: {role}) validated.")
    return True