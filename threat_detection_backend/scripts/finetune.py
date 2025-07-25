import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import yaml
from ultralytics import YOLO
import torch

# Configuration
PROJECT_DIR = Path("gun_detection_project")
TRAIN_DIR = Path("Train")  # Directory containing training data
TEST_DIR = Path("test")   # Directory containing test data
LABELS = ["weapon"]  # Single class for all weapons
TRAIN_VAL_SPLIT = 0.9  # 90% training, 10% validation from training data

def create_project_structure():
    """Create project directory structure"""
    # Create main directories
    dirs = [
        PROJECT_DIR,
        PROJECT_DIR / "data",
        PROJECT_DIR / "data" / "images" / "train",
        PROJECT_DIR / "data" / "images" / "val",
        PROJECT_DIR / "data" / "images" / "test",
        PROJECT_DIR / "data" / "labels" / "train",
        PROJECT_DIR / "data" / "labels" / "val",
        PROJECT_DIR / "data" / "labels" / "test",
        PROJECT_DIR / "weights",
        PROJECT_DIR / "results"
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    return True

def convert_bbox_to_yolo(size, box):
    """Convert VOC bbox to YOLO format"""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    
    # VOC format: xmin, ymin, xmax, ymax
    # YOLO format: x_center, y_center, width, height (normalized)
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    
    # Normalize
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    
    return x, y, w, h

def convert_annotation(xml_file, output_path, class_mapping):
    """Convert XML annotation to YOLO txt format"""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        with open(output_path, 'w') as out_file:
            for obj in root.iter('object'):
                cls = obj.find('name').text.lower()
                
                # Map any weapon-related class to our single "weapon" class
                if cls in ["weapon", "gun", "pistol", "rifle", "firearm", "handgun"]:
                    cls_id = 0  # Always use index 0 for the single "weapon" class
                else:
                    print(f"Warning: Unknown class '{cls}' in {xml_file}, skipping object")
                    continue
                
                xmlbox = obj.find('bndbox')
                b = (
                    float(xmlbox.find('xmin').text),
                    float(xmlbox.find('ymin').text),
                    float(xmlbox.find('xmax').text),
                    float(xmlbox.find('ymax').text)
                )
                
                # Convert to YOLO format
                bb = convert_bbox_to_yolo((width, height), b)
                
                # Write to output file
                out_file.write(f"{cls_id} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")
        
        return True
    except Exception as e:
        print(f"Error processing {xml_file}: {e}")
        return False

def prepare_dataset():
    """Prepare the dataset by converting annotations and organizing files"""
    # Process training data (with train/val split)
    train_files = process_directory(TRAIN_DIR, ["train", "val"])
    
    # Process test data (directly to test set)
    test_files = process_directory(TEST_DIR, ["test"])
    
    # Print dataset summary
    total_files = sum(train_files.values()) + sum(test_files.values())
    print(f"Total dataset files: {total_files}")
    print(f"Training files: {train_files.get('train', 0)}")
    print(f"Validation files: {train_files.get('val', 0)}")
    print(f"Test files: {test_files.get('test', 0)}")
    
    # Create data.yaml config file
    create_data_yaml()
    
    return train_files["train"], train_files["val"], test_files["test"]

def process_directory(source_dir, splits):
    """Process a directory (train or test) and distribute files to specified splits"""
    # Get all XML files in this directory
    annotation_files = list(Path(source_dir / "Annotations").glob("*.xml"))
    print(f"Found {len(annotation_files)} annotation files in {source_dir}")
    
    # Extract image filenames from annotations
    image_files = []
    for xml_file in annotation_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = root.find('filename').text
        
        # Handle the case where XML filename might not match the actual image filename
        img_file = Path(source_dir / "JPEGImages" / filename)
        if not img_file.exists():
            # Try matching by base name without extension
            potential_matches = list(Path(source_dir / "JPEGImages").glob(f"{Path(filename).stem}.*"))
            if potential_matches:
                img_file = potential_matches[0]
            else:
                # Try using the XML filename with .jpg extension
                img_file = Path(source_dir / "JPEGImages" / f"{xml_file.stem}.jpg")
        
        if img_file.exists():
            image_files.append((xml_file, img_file))
        else:
            print(f"Warning: No matching image found for {xml_file.name}")
    
    print(f"Successfully matched {len(image_files)} annotation-image pairs in {source_dir}")
    
    # Handle splits appropriately
    file_pairs_by_split = {}
    
    if "test" in splits and len(splits) == 1:
        # If this is test directory, all goes to test split
        file_pairs_by_split["test"] = image_files
    else:
        # If training directory, split into train/val
        train_pairs, val_pairs = train_test_split(
            image_files, train_size=TRAIN_VAL_SPLIT, random_state=42
        )
        file_pairs_by_split["train"] = train_pairs
        file_pairs_by_split["val"] = val_pairs
    
    # Process each split
    counts = {}
    for split_name, file_pairs in file_pairs_by_split.items():
        process_dataset_split(file_pairs, split_name)
        counts[split_name] = len(file_pairs)
    
    return counts

def process_dataset_split(file_pairs, split_name):
    """Process and copy files for a specific dataset split"""
    class_mapping = LABELS
    images_dir = PROJECT_DIR / "data" / "images" / split_name
    labels_dir = PROJECT_DIR / "data" / "labels" / split_name
    
    print(f"Processing {len(file_pairs)} files for {split_name} set")
    
    for xml_file, img_file in tqdm(file_pairs):
        # Copy image
        dest_img = images_dir / img_file.name
        shutil.copy(img_file, dest_img)
        
        # Convert and save annotation
        yolo_label = labels_dir / f"{xml_file.stem}.txt"
        convert_annotation(xml_file, yolo_label, class_mapping)

def create_data_yaml():
    """Create the data.yaml configuration file for YOLOv8"""
    # Use absolute paths instead of relative paths
    data = {
        'path': str(PROJECT_DIR.absolute() / "data"),  # Make path absolute
        'train': str((PROJECT_DIR.absolute() / "data" / "images" / "train")),  # Absolute path to train
        'val': str((PROJECT_DIR.absolute() / "data" / "images" / "val")),  # Absolute path to val
        'test': str((PROJECT_DIR.absolute() / "data" / "images" / "test")),  # Absolute path to test
        'names': {i: name for i, name in enumerate(LABELS)},
        'nc': len(LABELS)
    }
    
    with open(PROJECT_DIR / "data" / "dataset.yaml", 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"Created dataset configuration at {PROJECT_DIR / 'data' / 'dataset.yaml'}")

def train_model():
    """Train the YOLOv8 model with optimal settings"""
    print("Starting model training...")
    
    # Load YOLOv8 model
    model = YOLO('yolov8m.pt')  # Medium size for balance of speed and accuracy
    
    # Train the model with optimal hyperparameters
    results = model.train(
        data=str(PROJECT_DIR / "data" / "dataset.yaml"),
        epochs=100,
        patience=10,  # Early stopping
        batch=8,
        imgsz=640,
        pretrained=True,
        optimizer='AdamW',  # AdamW optimizer works well for detection tasks
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,  # L2 regularization to prevent overfitting
        warmup_epochs=3,
        cos_lr=True,  # Cosine learning rate schedule
        box=7.5,  # Box loss gain
        cls=0.5,  # Class loss gain
        dfl=1.5,  # Distribution focal loss gain
        val=True,
        plots=True,
        save=True,
        save_period=10,  # Save checkpoints every 10 epochs
        project=str(PROJECT_DIR / "results"),
        name='gun_detection',
        exist_ok=True,
        cache=False,  # Cache images for faster training
        device=0 if torch.cuda.is_available() else 'cpu',
        amp=True,  # Mixed precision for faster training
        augment=True,  # Use default augmentation
        mixup=0.1,  # Mix up augmentation
        mosaic=1.0,  # Mosaic augmentation
        degrees=0.3,  # Rotation augmentation (small for gun detection)
        translate=0.1,  # Translation augmentation
        scale=0.5,  # Scale augmentation
        shear=0.0,  # Shear augmentation (minimal for gun detection)
        perspective=0.0,  # Perspective augmentation (minimal for gun detection)
        flipud=0.0,  # No vertical flip for gun detection
        fliplr=0.5,  # Horizontal flip
        hsv_h=0.015,  # HSV hue augmentation
        hsv_s=0.7,  # HSV saturation augmentation 
        hsv_v=0.4,  # HSV value augmentation
    )
    
    return results

def export_model(format='onnx'):
    """Export the trained model to various formats"""
    # Get the best model
    best_model_path = list(Path(PROJECT_DIR / "results" / "gun_detection").glob('*.pt'))[0]
    model = YOLO(best_model_path)
    
    # Export the model
    model.export(format=format)
    print(f"Model exported to {format.upper()} format")

def run_inference(image_path):
    """Run inference on a single image"""
    # Get the best model
    best_model_path = list(Path(PROJECT_DIR / "results" / "gun_detection").glob('*.pt'))[0]
    model = YOLO(best_model_path)
    
    # Run inference
    results = model(image_path, conf=0.25)
    
    # Plot results
    for result in results:
        boxes = result.boxes
        print(f"Detected {len(boxes)} guns")
        
        # Plot the image with detections
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"Gun: {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.title("Gun Detection Results")
        plt.axis("off")
        plt.savefig(PROJECT_DIR / "results" / "inference_example.png")
        plt.close()
    
    return results

def verify_dataset():
    """Verify that label files contain data"""
    empty_files = 0
    total_files = 0
    
    for split in ["train", "val", "test"]:
        label_dir = PROJECT_DIR / "data" / "labels" / split
        if not label_dir.exists():
            continue
            
        label_files = list(label_dir.glob("*.txt"))
        total_files += len(label_files)
        
        for label_file in label_files:
            if label_file.stat().st_size == 0:
                empty_files += 1
    
    if empty_files > 0:
        print(f"⚠️ WARNING: Found {empty_files}/{total_files} empty label files!")
        print("Training will continue, treating empty label files as images without annotations.")
        return True
    else:
        print(f"✅ All {total_files} label files contain data")
        return True

def main():
    """Main execution function"""
    # Create project structure
    create_project_structure()
    
    # Prepare dataset
    #train_count, val_count, test_count = prepare_dataset()
    #print(f"Dataset prepared: {train_count} training samples, {val_count} validation samples, {test_count} test samples")
    
    # Verify dataset before training
    verify_dataset()  # Warn but do not abort on empty label files
    
    # Train model
    results = train_model()
    print("Training completed!")
    
    # Export model
    export_model(format='onnx')
    
    # Optional: Run inference on a test image
    test_images = list(Path(PROJECT_DIR / "data" / "images" / "test").glob("*.jpg"))
    if test_images:
        run_inference(str(test_images[0]))
        print(f"Inference example saved to {PROJECT_DIR / 'results' / 'inference_example.png'}")

if __name__ == "__main__":
    main()
