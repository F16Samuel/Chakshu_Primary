# For image inference
python infer.py --model gun_detection_project/results/gun_detection/weights/best.pt --source path/to/image.jpg

# For video inference
python infer.py --model gun_detection_project/results/gun_detection/weights/best.pt --source path/to/video.mp4 --output results.mp4

# For webcam
python infer.py --model gun_detection_project/results/gun_detection/weights/best.pt --source webcam
