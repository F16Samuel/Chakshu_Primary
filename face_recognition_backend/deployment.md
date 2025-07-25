To deploy and run your Campus Face Recognition System, follow these steps. The project structure seems to involve a `face` directory as the root, with `shared`, `check_server`, and `reg_server` subdirectories.

**Project Structure Overview (Assumed):**

```
/face
├── .env                  (Base .env file for shared/default configurations)
├── shared/
│   ├── __init__.py
│   ├── config.py         (Centralized configuration loading from .env)
│   ├── database.py
│   ├── face_processing.py (Contains face_recognition related utilities)
│   └── validation.py
├── reg_server/
│   ├── .env              (Environment variables specific to Registration server)
│   └── registration.py   (FastAPI application for Registration)
├── check_server/
│   ├── .env              (Environment variables specific to Recognition server)
│   └── recognition.py    (FastAPI application for Recognition)
└── (other project files like requirements.txt)
```

**Before you begin:**

  * **Install Python:** Ensure you have Python 3.8+ installed.
  * **Install Git:** If you plan to clone your repository.
  * **Create a Virtual Environment:** It's highly recommended to use a virtual environment to manage dependencies.

**Deployment Steps:**

1.  **Navigate to the Project Root:** Open your terminal or command prompt and navigate to the `face` directory, which contains `shared`, `reg_server`, `check_server`, and the base `.env` file.

    ```bash
    cd /path/to/your/face/project
    ```

2.  **Create and Activate a Virtual Environment:**

    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:** You'll need `FastAPI`, `uvicorn`, `python-dotenv`, `opencv-python`, and `face_recognition`. Ensure you have a `requirements.txt` file listing all your project dependencies. If not, you can create one.

    ```bash
    # Example requirements.txt content:
    # fastapi
    # uvicorn[standard]
    # python-dotenv
    # opencv-python
    # face_recognition
    # numpy

    pip install -r requirements.txt
    ```

      * **Note on `face_recognition`:** This library has system-level dependencies (like `dlib` and `CMake`). If `pip install face_recognition` fails, you might need to install these prerequisites manually for your operating system. Refer to the `face_recognition` GitHub page or common installation guides for your OS.

4.  **Database Initialization:** The `initialize_database()` function in your `shared/database.py` (which is called on server startup) will create the `campus_access.db` file if it doesn't exist. You don't need to manually create it, but ensure the directory where the DB will be created (specified by `DB_PATH` in `face/.env`, usually the `face` directory itself) is writable by the application.

5.  **Environment Variable Setup:**

      * Ensure you have the `.env` files in the following locations:
          * `face/.env` (base configuration)
          * `reg_server/.env` (registration server overrides)
          * `check_server/.env` (recognition server overrides)
      * Verify that the `DB_NAME`, `CARDS_DIR`, and `FACES_DIR` paths in the service-specific `.env` files are correctly set relative to their *own* folder, or relative to the `face` root if the application logic expects that. Given your provided `.env` files, `DB_NAME=../campus_access.db` for `reg_server` and `check_server` implies the database will be one level up, in the `face` directory. Similarly, `CARDS_DIR=../cards` and `FACES_DIR=../faces` in `reg_server/.env` would place these directories in the `face` folder.

**How to Run the Two Servers and Where to Run Them From:**

You should run each server from its respective directory to ensure that relative paths (like for `.env` files or data directories) are correctly resolved.

**To Run the Registration Server:**

1.  **Navigate to the `reg_server` directory:**
    ```bash
    cd /path/to/your/face/reg_server
    ```
2.  **Run the `registration.py` application:**
    ```bash
    uvicorn registration:app --host 0.0.0.0 --port 8000 --reload --log-level info
    ```
      * `registration:app`: Tells Uvicorn to look for an application named `app` inside `registration.py`.
      * `--host 0.0.0.0`: Makes the server accessible from any IP address (useful for external access).
      * `--port 8000`: Specifies the port number. This will be overridden by `PORT` in `reg_server/.env` if set.
      * `--reload`: (Optional, for development) Automatically reloads the server on code changes. Remove for production.
      * `--log-level info`: Sets the logging level. This will be overridden by `LOG_LEVEL` in `shared/config.py` (which in turn can be overridden by `.env` files).

**To Run the Recognition Server:**

1.  **Open a NEW terminal/command prompt.**
2.  **Navigate to the `check_server` directory:**
    ```bash
    cd /path/to/your/face/check_server
    ```
3.  **Run the `recognition.py` application:**
    ```bash
    uvicorn recognition:app --host 0.0.0.0 --port 8001 --reload --log-level info
    ```
      * `recognition:app`: Tells Uvicorn to look for an application named `app` inside `recognition.py`.
      * `--host 0.0.0.0`: Makes the server accessible from any IP address.
      * `--port 8001`: Specifies the port number. This will be overridden by `PORT` in `check_server/.env` if set.
      * `--reload`: (Optional, for development) Automatically reloads the server on code changes. Remove for production.
      * `--log-level info`: Sets the logging level.

**Important Considerations for Production Deployment (e.g., Render, Docker):**

  * **Dockerization:** For cloud deployments like Render, Heroku, AWS, etc., it's best practice to containerize your applications using Docker. You would create separate `Dockerfile`s for `reg_server` and `check_server` (or a multi-stage Dockerfile) and define how they are built and run.
  * **Process Manager:** Use a process manager like Gunicorn (with Uvicorn workers), PM2 (for Node.js, but also useful for managing Python processes), or systemd to ensure your applications run continuously, restart on crashes, and manage multiple worker processes.
  * **Environment Variables in Cloud:** Instead of `.env` files, cloud platforms provide mechanisms to set environment variables directly in their dashboards or via CLI commands. You would set `DB_NAME`, `CORS_ORIGINS`, `HOST`, `PORT`, etc., directly in the Render environment settings for each service.
  * **Persistent Storage:** For the `campus_access.db`, `cards`, and `faces` directories, you'll need persistent storage solutions provided by your cloud provider (e.g., Render Disks, AWS EFS, Google Cloud Filestore) if the data needs to persist beyond restarts. If your database is simple SQLite, it's often better to migrate to a managed database service like PostgreSQL or MySQL in production.
  * **Security:**
      * Configure `CORS_ORIGINS` to specific domains rather than `*` in production.
      * Implement proper authentication and authorization for your API endpoints.
      * Review all environment variables, especially `DEBUG_MODE` and sensitive settings.
  * **Hardware:** Ensure your deployment environment has sufficient CPU and RAM, especially for the face recognition service, which can be resource-intensive due to video processing and model inference.

By following these steps, you should be able to get your campus face recognition system up and running.