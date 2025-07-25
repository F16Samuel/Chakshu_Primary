import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import time

def run_inference_on_image(model_path, image_path, conf_threshold=0.5, save_path=None):
    """Run inference on a single image"""
    # Load model
    model = YOLO(model_path)
    
    # Run inference
    start_time = time.time()
    results = model(image_path, conf=conf_threshold)
    inference_time = time.time() - start_time
    
    # Process results
    img = cv2.imread(image_path)
    
    # Draw results on image
    for result in results:
        boxes = result.boxes
        print(f"Detected {len(boxes)} guns in {inference_time:.4f} seconds")
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"Gun: {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save or display the result
    if save_path:
        cv2.imwrite(save_path, img)
        print(f"Result saved to {save_path}")
    else:
        cv2.imshow("Gun Detection Result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def run_inference_on_video(model_path, video_path, conf_threshold=0.55, save_path=None):
    """Run inference on a video file"""
    # Load model
    model = YOLO(model_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create video writer if save_path is provided
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    
    # Process frames
    frame_count = 0
    total_time = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Start timing
        start_time = time.time()
        
        # Convert BGR to RGB and normalize
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference
        results = model(frame_rgb, conf=conf_threshold)
        
        # Calculate inference time
        inference_time = time.time() - start_time
        total_time += inference_time
        frame_count += 1
        
        # Draw results on frame
        annotated_frame = frame.copy()
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                # Filter out low-confidence detections
                if conf < conf_threshold:
                    continue
                
                # Draw bounding box and label
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Weapon: {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add FPS info
        fps_text = f"FPS: {1/inference_time:.1f}"
        cv2.putText(annotated_frame, fps_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Save or display the frame
        if save_path:
            writer.write(annotated_frame)
        else:
            cv2.imshow("Gun Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Release resources
    cap.release()
    if save_path:
        writer.release()
    cv2.destroyAllWindows()
    
    # Print statistics
    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"Processed {frame_count} frames in {total_time:.2f} seconds ({avg_fps:.2f} FPS)")

def run_inference_on_webcam(model_path, camera_id=0, conf_threshold=0.55):
    """Run inference on webcam"""
    # Load model
    model = YOLO(model_path)
    
    # Open webcam
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open webcam {camera_id}")
        return
    
    # Process frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Start timing
        start_time = time.time()
        
        # Convert BGR to RGB and normalize
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference
        results = model(frame_rgb, conf=conf_threshold)
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        # Draw results on frame
        annotated_frame = frame.copy()
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                # Filter out low-confidence detections
                if conf < conf_threshold:
                    continue
                
                # Draw bounding box and label
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Weapon: {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add FPS info
        fps_text = f"FPS: {1/inference_time:.1f}"
        cv2.putText(annotated_frame, fps_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow("Gun Detection (Press 'q' to quit)", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run inference with YOLOv8 gun detection model")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--source", type=str, required=True, 
                        help="Path to image, video file or 'webcam' for live detection")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--output", type=str, default=None, help="Path to save results")
    
    args = parser.parse_args()
    
    # Run inference based on source type
    if args.source.lower() == "webcam":
        run_inference_on_webcam(args.model, camera_id=0, conf_threshold=args.conf)
    elif args.source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        run_inference_on_video(args.model, args.source, conf_threshold=args.conf, save_path=args.output)
    else:
        run_inference_on_image(args.model, args.source, conf_threshold=args.conf, save_path=args.output)

if __name__ == "__main__":
    main()