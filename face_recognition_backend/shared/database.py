# shared/database.py
import sqlite3
import json
import logging
import numpy as np # Added for embedding handling in get_all_users
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Assuming DB_PATH and VALID_ROLES are defined in shared/config.py
# If not, you'll need to define them here or ensure config.py exists and is correct.
try:
    from shared.config import DB_PATH, VALID_ROLES
except ImportError:
    # Fallback for testing or if config.py isn't set up yet
    DB_PATH = Path("campus_access.db")
    VALID_ROLES = ['student', 'professor', 'guard', 'maintenance']
    logging.warning("shared/config.py not found or incomplete. Using default DB_PATH and VALID_ROLES.")


logger = logging.getLogger(__name__)

DATABASE_FILE = DB_PATH # Use the path from config

def get_db_connection():
    """Establishes and returns a database connection."""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        conn.row_factory = sqlite3.Row # This allows access to columns by name
        return conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        return None

def initialize_database() -> bool:
    """Initializes the database schema if it doesn't exist."""
    sql_create_users_table = """
    CREATE TABLE IF NOT EXISTS users (
        id_number TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        role TEXT NOT NULL,
        embedding TEXT NOT NULL, -- Stored as JSON string
        aadhar_path TEXT, -- Separate column for Aadhar path
        role_id_path TEXT, -- Separate column for Role ID path
        face_photo_paths TEXT, -- Stored as JSON string ['path1', 'path2']
        on_site INTEGER NOT NULL DEFAULT 0 -- SQLite uses INTEGER for BOOLEAN
    );
    """
    sql_create_access_log_table = """
    CREATE TABLE IF NOT EXISTS access_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        user_name TEXT NOT NULL,
        action TEXT NOT NULL, -- 'entry' or 'exit'
        timestamp TEXT NOT NULL, -- ISO 8601 format (e.g., 'YYYY-MM-DDTHH:MM:SSZ')
        method TEXT, -- 'scanner' or 'manual'
        confidence REAL, -- Optional: for scanner entries
        FOREIGN KEY (user_id) REFERENCES users (id_number)
    );
    """
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute(sql_create_users_table)
            cursor.execute(sql_create_access_log_table)
            conn.commit()
            logger.info("Database initialized successfully.")
            return True
        except sqlite3.Error as e:
            logger.error(f"Error initializing database: {e}")
            return False
        finally:
            conn.close()
    return False

# Modified save_user to accept aadhar_path and role_id_path separately
def save_user(role: str, name: str, id_number: str, embedding: List[float], aadhar_path: str, role_id_path: str, face_photo_paths: List[str]) -> bool:
    """Saves new user data to the database."""
    conn = get_db_connection()
    if conn:
        try:
            # Convert list to JSON strings for storage
            embedding_json = json.dumps(embedding)
            face_photo_paths_json = json.dumps(face_photo_paths)

            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (id_number, name, role, embedding, aadhar_path, role_id_path, face_photo_paths, on_site) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (id_number, name, role, embedding_json, aadhar_path, role_id_path, face_photo_paths_json, 0) # 0 for False
            )
            conn.commit()
            logger.info(f"User '{name}' (ID: {id_number}) saved to database.")
            return True
        except sqlite3.IntegrityError:
            logger.warning(f"User with ID '{id_number}' and role '{role}' already exists.")
            return False
        except sqlite3.Error as e:
            logger.error(f"Error saving user '{name}' (ID: {id_number}): {e}")
            return False
        finally:
            conn.close()
    return False

def check_user_exists(id_number: str) -> bool: # Removed 'role' from arguments for simpler check
    """Checks if a user with the given ID already exists (considering id_number as unique primary key)."""
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            # Changed query to just check id_number as it's the PRIMARY KEY
            cursor.execute("SELECT 1 FROM users WHERE id_number = ?", (id_number,))
            return cursor.fetchone() is not None
        except sqlite3.Error as e:
            logger.error(f"Error checking user existence for ID '{id_number}': {e}")
            return False
        finally:
            conn.close()
    return False

# Modified get_user_by_id to reflect new schema (no 'card_paths' to parse as JSON)
def get_user_by_id(id_number: str) -> Optional[Dict[str, Any]]:
    """Retrieves a user by their ID number."""
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE id_number = ?", (id_number,))
            user_data = cursor.fetchone()
            if user_data:
                user_dict = dict(user_data)
                # Parse JSON strings
                user_dict['embedding'] = json.loads(user_dict['embedding'])
                user_dict['face_photo_paths'] = json.loads(user_dict['face_photo_paths'])
                user_dict['on_site'] = bool(user_dict['on_site']) # Convert integer to boolean
                # No 'card_paths' to parse, aadhar_path and role_id_path are direct strings
                return user_dict
            return None
        except sqlite3.Error as e:
            logger.error(f"Error retrieving user by ID '{id_number}': {e}")
            return None
        finally:
            conn.close()
    return None

# Modified get_all_users to reflect new schema (no 'card_paths' to parse as JSON)
def get_all_users(role: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves all registered users, optionally filtered by role."""
    conn = get_db_connection()
    users = []
    if conn:
        try:
            cursor = conn.cursor()
            if role and role.lower() in VALID_ROLES:
                cursor.execute("SELECT * FROM users WHERE role = ?", (role.lower(),))
            else:
                cursor.execute("SELECT * FROM users")
            
            rows = cursor.fetchall()
            for row in rows:
                user_dict = dict(row)
                # Parse JSON strings back to Python objects
                try:
                    user_dict['embedding'] = np.array(json.loads(user_dict['embedding'])) # Use numpy array for embeddings
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Could not decode embedding for user {user_dict['id_number']}")
                    user_dict['embedding'] = np.array([]) # Default to empty numpy array
                
                try:
                    user_dict['face_photo_paths'] = json.loads(user_dict['face_photo_paths'])
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Could not decode face_photo_paths for user {user_dict['id_number']}")
                    user_dict['face_photo_paths'] = [] # Default to empty list
                
                user_dict['on_site'] = bool(user_dict['on_site']) # Convert integer to boolean

                users.append(user_dict)
            logger.debug(f"Retrieved {len(users)} users (filtered by role: {role if role else 'None'}).")
        except sqlite3.Error as e:
            logger.error(f"Error retrieving all users (role: {role}): {e}")
        finally:
            conn.close()
    return users

def update_user_status(id_number: str, on_site: bool) -> bool:
    """Updates the on_site status of a user."""
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            # SQLite uses 0 for False, 1 for True
            on_site_int = 1 if on_site else 0
            cursor.execute("UPDATE users SET on_site = ? WHERE id_number = ?", (on_site_int, id_number))
            conn.commit()
            if cursor.rowcount > 0:
                logger.info(f"User '{id_number}' status updated to on_site={on_site}.")
                return True
            else:
                logger.warning(f"No user found with ID '{id_number}' to update status.")
                return False
        except sqlite3.Error as e:
            logger.error(f"Error updating user status for ID '{id_number}': {e}")
            return False
        finally:
            conn.close()
    return False

def log_access_event(user_id: str, user_name: str, action: str, method: str, confidence: Optional[float] = None) -> bool:
    """Logs an access event (entry/exit) for a user."""
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            # Timestamp will be CURRENT_TIMESTAMP by default in DB, no need to pass it
            cursor.execute(
                "INSERT INTO access_logs (user_id, user_name, action, method, confidence) VALUES (?, ?, ?, ?, ?)",
                (user_id, user_name, action, method, confidence)
            )
            conn.commit()
            logger.info(f"Access event logged for {user_name} (ID: {user_id}), Action: {action}, Method: {method}.")
            return True
        except sqlite3.Error as e:
            logger.error(f"Error logging access event for user '{user_id}': {e}")
            return False
        finally:
            conn.close()
    return False

def get_access_logs(user_id: Optional[str] = None, action: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """Retrieves access logs, optionally filtered by user_id or action."""
    conn = get_db_connection()
    logs = []
    if conn:
        try:
            cursor = conn.cursor()
            query = "SELECT * FROM access_logs" # Corrected table name to access_logs
            params = []
            conditions = []

            if user_id:
                conditions.append("user_id = ?")
                params.append(user_id)
            if action:
                conditions.append("action = ?")
                params.append(action)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, tuple(params))
            
            rows = cursor.fetchall()
            for row in rows:
                logs.append(dict(row))
            logger.debug(f"Retrieved {len(logs)} access logs.")
        except sqlite3.Error as e:
            logger.error(f"Error retrieving access logs: {e}")
        finally:
            conn.close()
    return logs