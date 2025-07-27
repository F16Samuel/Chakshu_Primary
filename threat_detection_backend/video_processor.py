import cv2
import os
import uuid
import asyncio
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Dictionary to store processed video results
# Key: video_id, Value: List of detection results per frame
processed_video_results = {}

async def process_video_for_detections(video_path: str, detection_api, camera_id: str) -> str:
    """
    Processes a video file frame by frame using the detection API.

    Args:
        video_path (str): The path to the video file.
        detection_api: An instance of LiveWeaponDetectionAPI to run inference.
        camera_id (str): A unique identifier for this video processing task (e.g., "video_upload_uuid").

    Returns:
        str: A unique video_id under which the detection results are stored.
    """
    video_id = str(uuid.uuid4())
    processed_video_results[video_id] = []
    
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.error(f"Error: Could not open video file {video_path}")
        # Clean up the entry if video cannot be opened
        if video_id in processed_video_results:
            del processed_video_results[video_id]
        raise ValueError(f"Could not open video file: {video_path}")

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            # Run inference on the frame
            # The _run_inference method is expected to handle the image
            # and return detections.
            # We pass camera_id to differentiate video processing from live streams
            # and prevent live stream specific logging/state updates within _run_inference
            detection_result = detection_api._run_inference(frame, frame_id=frame_count, camera_id=camera_id)
            
            # Store the relevant parts of the detection result
            # Ensure the timestamp is an ISO string for consistency
            processed_video_results[video_id].append({
                "frame_id": frame_count,
                "timestamp": datetime.now().isoformat(), # Use current time for when it was processed
                "detections": detection_result.get("detections", [])
            })

            # For very large videos, you might want to yield or process in chunks,
            # but for simplicity, we'll process all frames and store in memory.
            # Consider memory implications for very long videos.

    except Exception as e:
        logger.error(f"Error during video processing for {video_path}: {e}")
        # Clean up the entry if an error occurs during processing
        if video_id in processed_video_results:
            del processed_video_results[video_id]
        raise
    finally:
        cap.release()
        # Clean up the temporary video file after processing
        if os.path.exists(video_path):
            os.remove(video_path)
            logger.info(f"Temporary video file {video_path} removed.")
        logger.info(f"Finished processing video {video_path}. Total frames: {frame_count}. Results stored under video_id: {video_id}")
    
    return video_id