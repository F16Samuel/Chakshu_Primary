# Function and API Documentation - Campus Access Control System

This document provides an overview of the core functions and API endpoints for the Campus Face Recognition Access Control System, segregated by their respective backend servers.

---

# Function and API Documentation - Campus Access Control System

This document provides an overview of the core functions and API endpoints for the Campus Face Recognition Access Control System, segregated by their respective backend servers.

---

## **1. Node.js Backend Server (`server.js`)**

This server acts as the primary data aggregation and dashboard interface, providing access to historical data and summaries from the SQLite database. It also serves static web assets and proxies requests to the Python microservices.

### **Core Functions**

* **Database Initialization & Connection:**
    * **Purpose:** Establishes a read-only connection to the `campus_access.db` SQLite database.
    * **Details:** Uses `sqlite3.OPEN_READONLY` to ensure data integrity and prevent accidental modifications from this server. Logs connection status.
* **`dbAll(sql, params)`**
    * **Purpose:** A helper function that promisifies `sqlite3.Database.all()`, allowing asynchronous execution of SQL queries that return multiple rows.
    * **Parameters:**
        * `sql`: The SQL query string.
        * `params`: An array of parameters for the SQL query.
    * **Returns:** A Promise that resolves with an array of rows, or rejects with an error.
* **`dbGet(sql, params)`**
    * **Purpose:** A helper function that promisifies `sqlite3.Database.get()`, allowing asynchronous execution of SQL queries that return a single row.
    * **Parameters:**
        * `sql`: The SQL query string.
        * `params`: An array of parameters for the SQL query.
    * **Returns:** A Promise that resolves with a single row object, or rejects with an error.
* **Static File Serving:**
    * **Purpose:** Configures Express to serve static web assets (HTML, CSS, JavaScript, images) from a designated `public` directory. This forms the user interface for the system's dashboard.
* **CORS Configuration:**
    * **Purpose:** Enables Cross-Origin Resource Sharing for all routes, allowing frontend applications (e.g., React apps running on a different port) to fetch data from this backend.
* **Graceful Shutdown:**
    * **Purpose:** Ensures the SQLite database connection is properly closed when the Node.js process receives a termination signal (e.g., Ctrl+C).

### **API Endpoints**

* **Static File Serving:**
    * **Purpose:** Serves static web assets (HTML, CSS, JavaScript, images) from the `public` directory. This is the main interface for users interacting with the system.
    * **Endpoints:** Handles requests for files like `/`, `/index.html`, `/entry.html`, `/exit.html`, `/register.html`, `/manual.html`, `/logs.html`.
* **`GET /api/activities`**
    * **Description:** Retrieves a list of recent access log activities. It joins the `access_logs` table with the `users` table to include the user's name in the activity record.
    * **Query:** `SELECT al.id, al.user_id, u.name AS userName, al.action, al.timestamp, al.method, al.confidence FROM access_logs al JOIN users u ON al.user_id = u.id_number ORDER BY al.timestamp DESC LIMIT 50`
    * **Response:** JSON array of activity objects.
* **`GET /api/personnel-breakdown`**
    * **Description:** Provides a breakdown of personnel currently on-site, categorized by role (students, professors, guards, maintenance).
    * **Query:** Multiple `SELECT COUNT(*)` queries on the `users` table, filtered by `role` and `on_site = 1`.
    * **Response:** JSON object with counts for each role.
* **`GET /api/total-on-site`**
    * **Description:** Returns the total number of personnel currently marked as on-site.
    * **Query:** `SELECT COUNT(*) FROM users WHERE on_site = 1`
    * **Response:** JSON object with `totalOnSite` count.
* **`GET /api/today-stats`**
    * **Description:** Provides daily activity statistics, including total entries, total exits, and the peak hour of activity for the current day.
    * **Query:** Multiple `SELECT COUNT(*)` and `SELECT STRFTIME('%H', timestamp)` queries on the `access_logs` table, filtered by the current day's timestamp range.
    * **Response:** JSON object with `totalEntries`, `entries`, `exits`, and `peakHour`.
* **Proxy Endpoints (to Python Recognition Server):**
    * **Purpose:** These endpoints forward requests to the Python recognition server (typically `http://localhost:8001`).
    * **`GET /api/recognition/health`**
        * **Description:** Checks the health status of the Python recognition API.
    * **`GET /api/recognition/logs`**
        * **Description:** Retrieves access logs from the Python recognition server. Supports optional query parameters for `user_id`, `action`, and `limit`.
    * **`GET /api/recognition/users/:user_id/status`**
        * **Description:** Retrieves the on-site status of a specific user by their ID.
    * **`POST /api/recognition/users/:user_id/manual_entry`**
        * **Description:** Manually logs a user's entry.
    * **`POST /api/recognition/users/:user_id/manual_exit`**
        * **Description:** Manually logs a user's exit.
    * **`POST /api/recognition/reload-faces`**
        * **Description:** Triggers the Python recognition server to reload known faces from the database.
* **Proxy Endpoints (to Python Registration Server):**
    * **Purpose:** These endpoints forward requests to the Python registration server (typically `http://localhost:8000`).
    * **`GET /api/registration/health`**
        * **Description:** Checks the health status of the Python registration API.
    * **`POST /api/registration/register`**
        * **Description:** Handles new user registration, forwarding the multipart form data directly.
    * **`POST /api/registration/webcam-capture`**
        * **Description:** Captures and saves a webcam photo (base64 encoded).
    * **`GET /api/registration/users/:user_id`**
        * **Description:** Retrieves detailed information for a specific registered user.
    * **`GET /api/registration/users`**
        * **Description:** Retrieves a list of all registered users.

---

## **2. Registration Server (`registration.py`)**

This FastAPI server handles user registration, including face photo uploads, ID card uploads, and face embedding generation.

### **Core Functions**

* **`create_directories()`**
    * **Purpose:** Ensures the existence of `CARDS_DIR` and `FACES_DIR` for storing uploaded files.
    * **Dependencies:** Uses `pathlib.Path.mkdir`.
* **`validate_file_type(filename)`**
    * **Purpose:** Checks if an uploaded file has a valid image extension (jpg, jpeg, png).
    * **Dependencies:** Uses `pathlib.Path.suffix`.
* **`save_uploaded_file(file, directory, prefix)`**
    * **Purpose:** Saves an `UploadFile` object to a specified directory with a unique filename.
    * **Dependencies:** Uses `os.path.join`, `pathlib.Path.stem`, `pathlib.Path.suffix`.
    * **Error Handling:** Raises `HTTPException` for invalid file types or saving errors.
* **`extract_face_encoding(image_path)`**
    * **Purpose:** Detects faces in an image and extracts the face encoding.
    * **Dependencies:** Uses `face_recognition.load_image_file`, `face_recognition.face_encodings`.
    * **Details:** Logs warnings if no face or multiple faces are found, returning the first encoding if multiple.
* **`generate_average_embedding(face_photo_paths)`**
    * **Purpose:** Calculates an average face embedding from a list of face photo paths.
    * **Dependencies:** Uses `extract_face_encoding` and `numpy.mean`.
* **`startup_event()`**
    * **Purpose:** FastAPI lifecycle event executed on application startup. It calls `create_directories()` and `shared.database.initialize_database()` to prepare the system.
* **`capture_webcam_photo(request)` (Helper for `/webcam-capture` endpoint)**
    * **Purpose:** Decodes a base64-encoded image and saves it to the `FACES_DIR`.

### **API Endpoints**

* **`GET /`**
    * **Description:** Root endpoint providing basic API information and available endpoints.
* **`GET /health`**
    * **Description:** Health check endpoint, returns a simple "API is running" status.
* **`POST /webcam-capture`**
    * **Description:** Accepts a base64 encoded image string (from webcam) and saves it to the `FACES_DIR` under the user's ID.
    * **Request Body:** `WebcamCaptureRequest` (Pydantic model with `user_id` and `image_data`).
    * **Response:** `UserRegistrationResponse`.
* **`POST /register`**
    * **Description:** Main user registration endpoint. Accepts user details (name, role, ID), Aadhar card, role ID card, and multiple face photos as multipart form data.
    * **Parameters (Form Data):** `name`, `role`, `id_number`, `aadhar_card` (UploadFile), `role_id_card` (UploadFile), `face_photo_1` (UploadFile), `face_photo_2` (UploadFile), `webcam_photo` (Optional UploadFile).
    * **Process:** Validates role and file sizes, checks for duplicate users, saves all uploaded files, generates an average face embedding, and then saves user data to the database via `shared.database.save_user`.
    * **Response:** `UserRegistrationResponse`.
    * **Dependencies (from `shared.database`):** `check_user_exists`, `save_user`.
* **`GET /users/{user_id}`**
    * **Description:** Retrieves specific user information by `id_number`.
    * **Parameters (Path):** `user_id`.
    * **Response:** JSON containing user details (ID, name, role, on_site status, Aadhar path, role ID path).
    * **Dependencies (from `shared.database`):** `get_user_by_id`.
* **`GET /users`**
    * **Description:** Retrieves a list of all registered users.
    * **Response:** JSON containing a list of user summaries (ID, name, role, on_site status, Aadhar path, role ID path) and total count.
    * **Dependencies (from `shared.database`):** `get_all_users`.

### **Dependencies (Functions from Shared Files)**

* **`shared.database`:**
    * `initialize_database()`: Called on startup to ensure database schema is set up.
    * `save_user()`: Used to store new user data (including embeddings and file paths) into the database.
    * `check_user_exists()`: Used during registration to prevent duplicate `id_number` entries.
    * `get_user_by_id()`: Used by `/users/{user_id}` endpoint.
    * `get_all_users()`: Used by `/users` endpoint.
* **`shared.config`:**
    * `CARDS_DIR`, `FACES_DIR`: Directories for storing uploaded files.
    * `VALID_ROLES`: List of allowed user roles for validation.
    * `MAX_FILE_SIZE`: Maximum allowed size for uploaded files.
    * `LOG_LEVEL`: Configures the logging verbosity.

---

## **3. Recognition Server (`recognition.py`)**

This FastAPI server handles real-time face recognition from webcams, updates user on-site status, and logs access events.

### **Core Functions**

* **`load_known_faces()`**
    * **Purpose:** Populates global lists (`known_face_encodings`, `known_face_names`, `known_user_ids`, `known_user_roles`) with data from all registered users in the database. These global lists are used for efficient face comparison during recognition.
    * **Dependencies:** Uses `shared.database.get_all_users()`.
* **`initialize_webcams()`**
    * **Purpose:** Initializes `cv2.VideoCapture` objects for entry and exit webcams based on `ENTRY_CAMERA_INDEX` and `EXIT_CAMERA_INDEX` from `config.py`.
    * **Details:** Includes fallback logic (`CAMERA_FALLBACK_ENABLED`) where the exit camera can use the entry camera index if its own fails.
    * **Dependencies:** Uses `cv2.VideoCapture`.
* **`startup_event()`**
    * **Purpose:** FastAPI lifecycle event executed on application startup. It calls `shared.database.initialize_database()`, `load_known_faces()`, and `initialize_webcams()`.
* **`shutdown_event()`**
    * **Purpose:** FastAPI lifecycle event executed on application shutdown. It properly releases both entry and exit webcam resources (`cv2.VideoCapture.release()`).
* **`entry_websocket_endpoint()` (within `while True` loop)**
    * **Purpose:** Reads frames from the entry webcam, performs face detection and recognition, draws overlays, updates user status, logs events, and prepares processed frames for sending.
    * **Process:** Resizes frame for faster face detection (`FACE_DETECTION_SCALE`), uses `face_recognition.face_locations` and `face_recognition.face_encodings`, compares against `known_face_encodings` using `EMBEDDING_THRESHOLD`, and applies a `COOLDOWN_PERIOD_SECONDS` to prevent rapid duplicate events for the same user.
    * **Drawing:** Uses `cv2.rectangle` and `cv2.putText` for visual feedback, applying `UNKNOWN_COLOR` or `ROLE_COLORS` based on recognition results.
    * **Dependencies:** `face_recognition` library, `cv2` (OpenCV), `shared.database.update_user_status`, `shared.database.log_access_event`.
* **`exit_websocket_endpoint()` (within `while True` loop)**
    * **Purpose:** Similar to `entry_websocket_endpoint`, but for the exit webcam. It marks users as "off-site."
    * **Dependencies:** Same as `entry_websocket_endpoint`.

### **API Endpoints**

* **`GET /`**
    * **Description:** Root endpoint providing basic API information and available endpoints.
* **`GET /health`**
    * **Description:** Health check endpoint, returns a simple "API is running" status.
* **`GET /reload-faces`**
    * **Description:** Manually triggers the `load_known_faces()` function to refresh the in-memory cache of known faces from the database. Useful after new registrations on the registration server.
* **`GET /logs`**
    * **Description:** Retrieves access log entries from the database.
    * **Parameters (Query):** `user_id` (optional), `action` (optional, 'entry'/'exit'), `limit` (optional, default 100).
    * **Dependencies (from `shared.database`):** `get_access_logs`.
* **`GET /users/{user_id}/status`**
    * **Description:** Retrieves the current on-site/off-site status of a specific user.
    * **Parameters (Path):** `user_id`.
    * **Dependencies (from `shared.database`):** `get_user_by_id`.
* **`POST /users/{user_id}/manual_entry`**
    * **Description:** Manually logs a user's entry event and updates their status to on-site.
    * **Parameters (Path):** `user_id`.
    * **Dependencies (from `shared.database`):** `get_user_by_id`, `update_user_status`, `log_access_event`.
* **`POST /users/{user_id}/manual_exit`**
    * **Description:** Manually logs a user's exit event and updates their status to off-site.
    * **Parameters (Path):** `user_id`.
    * **Dependencies (from `shared.database`):** `get_user_by_id`, `update_user_status`, `log_access_event`.

### **Dependencies (Functions from Shared Files)**

* **`shared.database`:**
    * `initialize_database()`: Called on startup to ensure database schema is set up.
    * `get_all_users()`: Used by `load_known_faces()` to retrieve user data.
    * `update_user_status()`: Used to set a user's `on_site` status during recognition.
    * `log_access_event()`: Records entry/exit events.
    * `get_access_logs()`: Used by `/logs` endpoint.
    * `get_user_by_id()`: Used by `/users/{user_id}/status` and manual entry/exit.
* **`shared.config`:**
    * `LOG_LEVEL`: Configures logging verbosity.
    * `RECOGNITION_HOST`, `RECOGNITION_PORT`: Server host and port.
    * `ENTRY_CAMERA_INDEX`, `EXIT_CAMERA_INDEX`: Camera indices for OpenCV.
    * `CAMERA_FALLBACK_ENABLED`: Boolean for camera fallback logic.
    * `EMBEDDING_THRESHOLD`: Threshold for face recognition confidence.
    * `FACE_DETECTION_SCALE`: Scale factor for reducing frame size during face detection.
    * `COOLDOWN_PERIOD_SECONDS`: Cooldown to prevent rapid duplicate detections.
    * `CORS_ORIGINS`, `CORS_CREDENTIALS`: CORS settings for the FastAPI app.
    * `UNKNOWN_COLOR`, `ROLE_COLORS`: Color definitions for drawing bounding boxes on video streams.

---

## **4. Shared Utility Files**

These files contain common functions and configurations used across multiple services.

### **`shared/database.py`**

* **Purpose:** Provides a centralized interface for all database interactions using SQLite.
* **Core Functions:**
    * **`get_db_connection()`**: Establishes a connection to the `campus_access.db` file, ensuring `sqlite3.Row` factory is set for named column access.
    * **`initialize_database()`**: Creates the `users` and `access_logs` tables with their defined schemas if they do not already exist.
    * **`save_user(role, name, id_number, embedding, aadhar_path, role_id_path, face_photo_paths)`**: Inserts a new user record into the `users` table. `embedding` and `face_photo_paths` are stored as JSON strings.
    * **`check_user_exists(id_number)`**: Checks if a user with the given `id_number` (primary key) already exists.
    * **`get_user_by_id(id_number)`**: Retrieves a single user's data by their `id_number`, parsing JSON fields back into Python objects.
    * **`get_all_users(role=None)`**: Retrieves all registered users, optionally filtered by role. Parses JSON fields and converts `embedding` to `numpy.array`.
    * **`update_user_status(id_number, on_site)`**: Updates the `on_site` boolean status for a given user.
    * **`log_access_event(user_id, user_name, action, method, confidence)`**: Records an access event in the `access_logs` table.
    * **`get_access_logs(user_id=None, action=None, limit=100)`**: Fetches access log entries, with optional filtering by `user_id` or `action`, and a limit on results.

### **`shared/config.py`**

* **Purpose:** Centralized configuration management for the entire system, loading settings from environment variables and providing defaults.
* **Key Settings Defined:**
    * **Database:** `DB_NAME`, `DB_PATH`.
    * **Directory Paths:** `CARDS_DIR`, `FACES_DIR`.
    * **Face Recognition:** `EMBEDDING_THRESHOLD`, `RECOGNITION_FPS`, `STREAM_FPS`, `FACE_DETECTION_SCALE`, `FACE_PROCESSING_INTERVAL`.
    * **File Uploads:** `MAX_FILE_SIZE`, `VALID_EXTENSIONS`, `ALLOWED_MIME_TYPES`.
    * **Server Configuration:** `REGISTRATION_HOST`, `REGISTRATION_PORT`, `RECOGNITION_HOST`, `RECOGNITION_PORT`.
    * **Camera Configuration:** `ENTRY_CAMERA_INDEX`, `EXIT_CAMERA_INDEX`, `CAMERA_FALLBACK_ENABLED`, `FRAME_TIMEOUT`, `CAMERA_RETRY_ATTEMPTS`, `CAMERA_RETRY_DELAY`.
    * **Security:** `CORS_ORIGINS`, `CORS_CREDENTIALS`.
    * **Logging:** `LOG_LEVEL`, `DEBUG_MODE`.
    * **Roles:** `VALID_ROLES`.
    * **Registration Specifics:** `REQUIRE_WEBCAM_PHOTO`, `MIN_FACE_PHOTOS`, `MAX_FACE_PHOTOS`.
    * **API Meta:** `API_TITLE`, `API_DESCRIPTION`, `API_VERSION`.
    * **Scanner Control:** `DEFAULT_ENTRY_SCANNING`, `DEFAULT_EXIT_SCANNING`.
    * **Colors:** `UNKNOWN_COLOR`, `ROLE_COLORS` (BGR format).
    * **Performance:** `MAX_CONCURRENT_STREAMS`, `MEMORY_CLEANUP_INTERVAL`.
    * **Rate Limiting:** `ENABLE_RATE_LIMITING`, `MAX_REQUESTS_PER_MINUTE`.

### **`shared/face_processing.py`**

* **Purpose:** (Based on common project structure) This file would typically contain reusable functions related to face detection, encoding, or image manipulation that are shared between `registration.py` and `recognition.py`.
* **Current status:** In the provided code, the core face processing functions (`extract_face_encoding`, `generate_average_embedding`) are currently defined within `registration.py`. If this file were intended to hold those, they would be moved here. However, based on the provided `registration.py` and `recognition.py` content, its functions are *not directly used* by the current `recognition.py` (it uses `face_recognition` directly and `registration.py` contains its own helper for embeddings).

### **`shared/validation.py`**

* **Purpose:** (Based on common project structure) This file would typically contain reusable validation logic, e.g., for user input, file types, or specific data formats, shared between services.
* **Current status:** In the provided code, file type validation (`validate_file_type`) is currently defined within `registration.py`. If this file were intended to hold those, they would be moved here. However, based on the provided code, its functions are *not directly used* by the current `recognition.py` or `registration.py` (which implement validation inline or in helpers).

###   **`shared/__init__.py`**

* **Purpose:** Marks the `shared` directory as a Python package, allowing its modules (like `database` and `config`) to be imported by other parts of the application (e.g., `from shared.database import ...`).
* **Current status:** Essential for the Python import structure.

---

## **Key Features & Capabilities (Across the System)**

### **Advanced Access Control**
- **Dual-camera architecture** with independent entry/exit monitoring (via `ENTRY_CAMERA_INDEX`, `EXIT_CAMERA_INDEX`, `entry_webcam`, `exit_webcam` in `recognition.py` and proxied by `server.js`).
- **State-aware processing** prevents duplicate entries/exits using `last_entry_detection_time` and `last_exit_detection_time` with `COOLDOWN_PERIOD_SECONDS`.
- **Automatic status synchronization** updates database immediately upon successful face recognition using `shared.database.update_user_status`.
- **Role-based visual coding** with distinct colors for different personnel types (`ROLE_COLORS` from `config.py` applied in `recognition.py` video streams).
- **Manual Access Control** allows guards to manually log entries and exits via dedicated API endpoints (`/manual_entry`, `/manual_exit` in `recognition.py`, proxied by `server.js`).

### **Performance & Reliability**
- **Configurable `FACE_DETECTION_SCALE`** for optimal CPU usage during face detection.
- **Automatic camera recovery** with fallback mechanisms (`CAMERA_FALLBACK_ENABLED` in `config.py`) when webcam connections fail (`initialize_webcams()` in `recognition.py`).
- **Comprehensive error handling** with detailed `logging` for debugging and system monitoring across all services.

### **Security & Monitoring**
- **Confidence-based recognition** with adjustable `EMBEDDING_THRESHOLD` and percentage scoring (`1 - face_distance`).
- **Complete access logging** with timestamps, methods (`scanner`/`manual`), and confidence levels via `shared.database.log_access_event`.
- **Real-time visual feedback** showing recognition status, access decisions, and scanner activity states.
- **Fail-safe unknown person handling** with `UNKNOWN_COLOR` applied to unrecognized faces in video streams.
- **CORS Configuration** (via `CORS_ORIGINS`, `CORS_CREDENTIALS` in `config.py`) to control cross-origin access.

### **User Registration & Management**
- **Dedicated Registration API** (`registration.py`) for secure user onboarding.
- **Support for multiple face photos** (`face_photo_1`, `face_photo_2`, `webcam_photo`) to generate a robust average embedding.
- **Storage of Aadhar and Role ID cards** alongside user data for record-keeping.
- **Duplicate user prevention** during registration (`shared.database.check_user_exists`).
- **Centralized User Database** (`shared.database.py`) provides functions for saving, retrieving, and updating user information.