# shared/face_processing.py
import face_recognition
import numpy as np
import logging
from typing import List, Optional, Tuple, Union
import cv2

logger = logging.getLogger(__name__)

def extract_face_encoding(image_path: str) -> Optional[np.ndarray]:
    """
    Extracts the face encoding from an image file.
    Expects a single face in the image for registration purposes.
    """
    try:
        # Load the image using face_recognition (uses PIL internally)
        image = face_recognition.load_image_file(image_path)
        
        # Find all face encodings in the image
        face_encodings = face_recognition.face_encodings(image)

        if not face_encodings:
            logger.warning(f"No face found in image: {image_path}")
            return None
        elif len(face_encodings) > 1:
            logger.warning(f"Multiple faces found in image: {image_path}. Using the first one.")
            return face_encodings[0]
        else:
            return face_encodings[0]
    except FileNotFoundError:
        logger.error(f"Image file not found: {image_path}")
        return None
    except Exception as e:
        logger.error(f"Error processing image {image_path} for face encoding: {e}")
        return None

def extract_face_encodings_from_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Extracts the face encoding from image bytes.
    Expects a single face in the image for registration purposes.
    """
    try:
        # Convert bytes to a numpy array, then to a face_recognition image
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # Use OpenCV to decode
        
        if image is None:
            logger.error("Could not decode image bytes. Invalid image data.")
            return None

        # Convert BGR (OpenCV) to RGB (face_recognition)
        rgb_image = image[:, :, ::-1] 
        
        face_encodings = face_recognition.face_encodings(rgb_image)

        if not face_encodings:
            logger.warning("No face found in image bytes.")
            return None
        elif len(face_encodings) > 1:
            logger.warning("Multiple faces found in image bytes. Using the first one.")
            return face_encodings[0]
        else:
            return face_encodings[0]
    except Exception as e:
        logger.error(f"Error processing image bytes for face encoding: {e}")
        return None

def extract_face_locations_and_encodings_from_frame(frame: np.ndarray, scale_factor: float = 0.25) -> Tuple[List[Tuple[int, int, int, int]], List[np.ndarray]]:
    """
    Detects face locations and extracts encodings from a single video frame.
    Scales down the image for faster processing, then scales back coordinates.
    """
    if frame is None:
        return [], []

    # Resize frame for faster face detection
    small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # Scale back up face locations to match the original frame size
    scaled_face_locations = []
    for top, right, bottom, left in face_locations:
        top = int(top / scale_factor)
        right = int(right / scale_factor)
        bottom = int(bottom / scale_factor)
        left = int(left / scale_factor)
        scaled_face_locations.append((top, right, bottom, left))

    return scaled_face_locations, face_encodings

def compare_faces(known_encodings: List[np.ndarray], face_encoding_to_check: np.ndarray, tolerance: float) -> Tuple[bool, Optional[int], float]:
    """
    Compares a single face encoding to a list of known encodings.
    Returns (is_match, match_index, min_distance).
    """
    if not known_encodings:
        return False, None, float('inf')

    # Compute distances to all known faces
    face_distances = face_recognition.face_distance(known_encodings, face_encoding_to_check)
    
    # Find the best match
    min_distance = np.min(face_distances)
    
    # Check if the best match is within tolerance
    if min_distance < tolerance:
        match_index = np.argmin(face_distances)
        return True, match_index, min_distance
    else:
        return False, None, min_distance

def generate_average_embedding(encodings: List[np.ndarray]) -> Optional[np.ndarray]:
    """
    Generates an average embedding from a list of face encodings.
    Useful for creating a robust single embedding from multiple photos of the same person.
    """
    if not encodings:
        logger.warning("No encodings provided to generate average embedding.")
        return None
    
    # Stack all encodings vertically and compute the mean along the rows
    average_embedding = np.mean(encodings, axis=0)
    return average_embedding

def calculate_confidence(distance: float, threshold: float = 0.6) -> float:
    """
    Calculates a confidence score based on the face distance.
    Higher confidence for smaller distances. Scales from 100% (distance 0) to 0% (at or above threshold).
    """
    if distance >= threshold:
        return 0.0
    # Linear scaling: confidence = (1 - distance/threshold) * 100
    confidence = (1 - (distance / threshold)) * 100
    return max(0.0, min(100.0, confidence)) # Ensure bounds [0, 100]

def draw_face_bounding_box(frame: np.ndarray, face_location: Tuple[int, int, int, int], label: str, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Draws a bounding box and label around a detected face in a video frame.

    Args:
        frame (np.ndarray): The image frame (OpenCV BGR format).
        face_location (Tuple[int, int, int, int]): A tuple (top, right, bottom, left)
                                                    representing the face location.
        label (str): The text label to display (e.g., "Name (Role) - Confidence%").
        color (Tuple[int, int, int]): BGR color tuple for the box and text (default green).

    Returns:
        np.ndarray: The frame with the bounding box and label drawn.
    """
    top, right, bottom, left = face_location

    # Draw a box around the face
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    # Draw a label with a name below the face
    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, label, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1) # White text

    return frame
