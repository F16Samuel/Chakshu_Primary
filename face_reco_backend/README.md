# Face Recognition Backend - Setup & Usage Guide

## Overview
The Chakshu Face Recognition System is a FastAPI-based backend service that provides campus access control using facial recognition technology. It integrates with MongoDB for data storage and includes features for personnel registration, face recognition, and activity tracking.

## Prerequisites

### System Requirements
- Python 3.8 or higher
- MongoDB 4.4 or higher
- Minimum 500MB RAM
- 10GB free disk space
- Camera/webcam for face capture

### Required Libraries and Dependencies
```bash
# Core dependencies
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
motor>=3.2.0
pymongo>=4.4.0
python-multipart>=0.0.6

# Face recognition dependencies
face-recognition>=1.3.0
opencv-python>=4.8.0
numpy>=1.24.0
dlib>=19.24.0

# Additional utilities
python-dotenv>=1.0.0
Pillow>=10.0.0
aiofiles>=23.0.0
```

## Installation

### Step 1: Clone and Setup Project
```bash
# Create project directory
mkdir chakshu-face-recognition
cd chakshu-face-recognition

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 2: Install Dependencies
```bash
# Install Python packages
pip install fastapi uvicorn[standard] motor pymongo python-multipart
pip install face-recognition opencv-python numpy dlib
pip install python-dotenv Pillow aiofiles

# For face-recognition installation issues on some systems:
# Ubuntu/Debian:
sudo apt-get install build-essential cmake
sudo apt-get install libopenblas-dev liblapack-dev
sudo apt-get install libx11-dev libgtk-3-dev

# macOS:
brew install cmake
brew install dlib
```

### Step 3: MongoDB Setup

#### Option A: Local MongoDB Installation
```bash
# Ubuntu/Debian
sudo apt-get install mongodb

# macOS
brew install mongodb-community

# Start MongoDB service
sudo systemctl start mongodb  # Linux
brew services start mongodb-community  # macOS
```

#### Option B: MongoDB Cloud (Atlas)
1. Go to [MongoDB Atlas](https://cloud.mongodb.com)
2. Create a free cluster
3. Create a database user
4. Get connection string

### Step 4: Environment Configuration
Create a `.env` file in your project root:

```env
# Server Configuration
HOST=0.0.0.0
PORT=8000

# MongoDB Configuration
MONGO_URI=mongodb://localhost:27017  # For local MongoDB
# MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net  # For Atlas
MONGO_DB_NAME=campus_access

# Directory Configuration
CARDS_DIR=./uploads/cards
FACES_DIR=./uploads/faces

# Role Configuration
VALID_ROLES=["student", "professor", "guard", "maintenance"]
ROLE_COLORS={"student": [0, 255, 0], "professor": [255, 0, 0], "guard": [255, 0, 255], "maintenance": [0, 255, 255]}
UNKNOWN_COLOR=[0, 165, 255]
```

### Step 5: Create Directory Structure
```bash
mkdir -p uploads/cards
mkdir -p uploads/faces
mkdir -p logs
```

### Step 6: Save the Backend Code
Save the provided Python code as `main.py` in your project directory.

## Running the Application

### Development Mode
```bash
# Start the server
python main.py

# Or using uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Production Mode
```bash
# Install production server
pip install gunicorn

# Run with Gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Verify Installation
1. Open browser and go to `http://localhost:8000/health`
2. You should see: `{"status": "healthy", "timestamp": "...", "database": "connected"}`
3. API documentation available at: `http://localhost:8000/docs`

## API Endpoints Usage

### 1. Health Check
```bash
curl -X GET "http://localhost:8000/health"
```

### 2. Register Personnel
```bash
curl -X POST "http://localhost:8000/register" \
  -F "name=John Doe" \
  -F "role=student" \
  -F "id_number=STU001" \
  -F "aadhar_card=@/path/to/aadhar.jpg" \
  -F "role_id_card=@/path/to/student_id.jpg" \
  -F "face_photos=@/path/to/face1.jpg" \
  -F "face_photos=@/path/to/face2.jpg"
```

### 3. Face Recognition
```bash
curl -X POST "http://localhost:8000/recognize_face" \
  -F "file=@/path/to/test_image.jpg"
```

### 4. Get Users List
```bash
curl -X GET "http://localhost:8000/users"
```

### 5. Update User Status
```bash
curl -X PUT "http://localhost:8000/users/USER_ID" \
  -H "Content-Type: application/json" \
  -d '{"status": "entry"}'
```

### 6. Dashboard APIs
```bash
# Recent activities
curl -X GET "http://localhost:8000/dashboard/activities"

# Personnel breakdown
curl -X GET "http://localhost:8000/dashboard/personnel-breakdown"

# Total on-site
curl -X GET "http://localhost:8000/dashboard/total-on-site"

# Today's statistics
curl -X GET "http://localhost:8000/dashboard/today-stats"
```

## Frontend Integration

### JavaScript Example
```javascript
// Register new personnel
async function registerPersonnel(formData) {
    const response = await fetch('http://localhost:8000/register', {
        method: 'POST',
        body: formData
    });
    return await response.json();
}

// Recognize face
async function recognizeFace(imageFile) {
    const formData = new FormData();
    formData.append('file', imageFile);
    
    const response = await fetch('http://localhost:8000/recognize_face', {
        method: 'POST',
        body: formData
    });
    return await response.json();
}

// Get dashboard data
async function getDashboardData() {
    const [activities, breakdown, onSite, stats] = await Promise.all([
        fetch('http://localhost:8000/dashboard/activities').then(r => r.json()),
        fetch('http://localhost:8000/dashboard/personnel-breakdown').then(r => r.json()),
        fetch('http://localhost:8000/dashboard/total-on-site').then(r => r.json()),
        fetch('http://localhost:8000/dashboard/today-stats').then(r => r.json())
    ]);
    
    return { activities, breakdown, onSite, stats };
}
```

## Database Schema

### Users Collection
```json
{
  "_id": "ObjectId",
  "name": "John Doe",
  "role": "student",
  "id_number": "STU001",
  "on_site": false,
  "registered_at": "2024-01-01T00:00:00Z",
  "last_activity": "2024-01-01T10:30:00Z",
  "face_encodings": [[0.1, 0.2, ...], [0.3, 0.4, ...]]
}
```

### Activities Collection
```json
{
  "_id": "ObjectId",
  "user_id": "user_object_id",
  "name": "John Doe",
  "role": "student",
  "action": "entry",
  "confidence": 0.85,
  "timestamp": "2024-01-01T10:30:00Z",
  "type": "recognition"
}
```

## Troubleshooting

### Common Issues

#### 1. Face Recognition Library Installation
```bash
# If face_recognition fails to install:
pip install cmake
pip install dlib
pip install face_recognition

# On Windows, try:
pip install face_recognition --no-cache-dir
```

#### 2. MongoDB Connection Issues
```bash
# Check MongoDB status
sudo systemctl status mongodb

# Restart MongoDB
sudo systemctl restart mongodb

# Check logs
sudo journalctl -u mongodb
```

#### 3. Import Errors
```bash
# Install missing dependencies
pip install -r requirements.txt

# Check Python version
python --version  # Should be 3.8+
```

#### 4. File Upload Issues
- Ensure upload directories exist and have write permissions
- Check file size limits (default: 16MB for MongoDB GridFS)
- Verify image formats are supported (JPG, PNG, etc.)

### Performance Optimization

#### 1. Face Recognition Optimization
```python
# Adjust tolerance for accuracy vs speed
face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)

# Reduce image size for faster processing
def resize_image(image, max_width=800):
    height, width = image.shape[:2]
    if width > max_width:
        ratio = max_width / width
        new_height = int(height * ratio)
        return cv2.resize(image, (max_width, new_height))
    return image
```

#### 2. Database Optimization
- Ensure proper indexes are created (handled automatically)
- Use connection pooling for high-traffic scenarios
- Consider caching frequently accessed data

## Security Considerations

### 1. Environment Variables
- Never commit `.env` file to version control
- Use strong MongoDB credentials
- Implement API authentication for production

### 2. File Upload Security
```python
# Add file type validation
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
```

### 3. Production Deployment
- Use HTTPS in production
- Implement rate limiting
- Add input validation and sanitization
- Use environment-specific configurations

## Monitoring and Logging

### Log Configuration
```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
```

### Health Monitoring
- Monitor `/health` endpoint
- Set up alerts for database connectivity
- Track face recognition accuracy rates
- Monitor disk usage for uploaded files

## Support and Maintenance

### Regular Maintenance Tasks
1. **Database Cleanup**: Remove old activity logs periodically
2. **File Management**: Clean up orphaned uploaded files
3. **Performance Monitoring**: Check response times and accuracy
4. **Backup**: Regular database backups
5. **Updates**: Keep dependencies updated for security

### Getting Help
- Check API documentation at `/docs` endpoint
- Review application logs in `logs/` directory
- Monitor database performance using MongoDB tools
- Test individual components using the provided API endpoints

This comprehensive guide should help you set up and run the face recognition backend service successfully. Make sure to test each component thoroughly before deploying to production.