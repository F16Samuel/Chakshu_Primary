# mongo_database.py
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure, OperationFailure
import numpy as np # For handling numpy arrays if embeddings are stored as such
import json # Used for parsing VALID_ROLES from environment variable

logger = logging.getLogger(__name__)

# Environment variables for MongoDB connection
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "campus_access")

# Load VALID_ROLES directly from environment for consistent dashboard breakdown
# This needs to be parsed as JSON as it's a list string in .env
VALID_ROLES_STR = os.getenv("VALID_ROLES", '["student", "professor", "guard", "maintenance"]')
try:
    VALID_ROLES = json.loads(VALID_ROLES_STR)
except json.JSONDecodeError:
    logger.error(f"Failed to parse VALID_ROLES from environment: {VALID_ROLES_STR}. Using default.")
    VALID_ROLES = ["student", "professor", "guard", "maintenance"]


class MongoDB:
    _client: Optional[AsyncIOMotorClient] = None
    _db = None
    _instance = None # For singleton pattern

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MongoDB, cls).__new__(cls)
            # We don't call _connect here directly in __new__ because it's async.
            # Connection will be established via get_mongo_db() or explicit connect call.
        return cls._instance

    async def connect(self):
        """Establishes an asynchronous connection to MongoDB."""
        if self._client is None:
            try:
                self._client = AsyncIOMotorClient(self.MONGO_URI_CLASS_VAR) # Use a class variable for URI
                await self._client.admin.command('ping') # Test connection
                self._db = self._client[self.MONGO_DB_NAME_CLASS_VAR] # Use a class variable for DB name
                logger.info(f"Successfully connected to MongoDB: {self.MONGO_DB_NAME_CLASS_VAR}")
            except ConnectionFailure as e:
                logger.error(f"MongoDB connection failed: {e}")
                self._client = None
                self._db = None
                raise
            except Exception as e:
                logger.error(f"An unexpected error occurred during MongoDB connection: {e}")
                self._client = None
                self._db = None
                raise

    def get_db(self):
        """Returns the MongoDB database object."""
        if self._db is None:
            # In an async context, you should await connect() before calling get_db()
            # This method should ideally only be called after connect() has succeeded.
            raise ConnectionError("MongoDB is not connected. Call await get_mongo_db() first.")
        return self._db

    async def close_connection(self):
        """Closes the MongoDB connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            logger.info("MongoDB connection closed.")

# Initialize MongoDB connection parameters within the class for clarity
MongoDB.MONGO_URI_CLASS_VAR = MONGO_URI
MongoDB.MONGO_DB_NAME_CLASS_VAR = MONGO_DB_NAME

async def get_mongo_db():
    """Helper function to get the MongoDB database instance, ensuring connection."""
    mongo_instance = MongoDB()
    if mongo_instance._client is None: # Check if client is not yet established
        await mongo_instance.connect()
    return mongo_instance.get_db()

# --- User Management Functions ---

async def save_user(user_data: Dict[str, Any]) -> bool:
    """Saves a new user or updates an existing one in MongoDB."""
    db = await get_mongo_db()
    users_collection = db['users']
    try:
        # Convert numpy array to list for MongoDB storage
        if 'face_embedding' in user_data and isinstance(user_data['face_embedding'], np.ndarray):
            user_data['face_embedding'] = user_data['face_embedding'].tolist()

        # Update if user_id exists, otherwise insert
        result = await users_collection.update_one(
            {'id_number': user_data['id_number']},
            {'$set': user_data},
            upsert=True
        )
        if result.upserted_id:
            logger.info(f"User '{user_data['id_number']}' registered successfully.")
            return True
        elif result.modified_count > 0:
            logger.info(f"User '{user_data['id_number']}' updated successfully.")
            return True
        else:
            logger.warning(f"User '{user_data['id_number']}' not inserted or updated.")
            return False
    except OperationFailure as e:
        logger.error(f"MongoDB operation failed during save_user: {e}")
        return False
    except Exception as e:
        logger.error(f"Error saving user '{user_data.get('id_number', 'N/A')}': {e}")
        return False

async def check_user_exists(id_number: str) -> bool:
    """Checks if a user with the given ID number exists."""
    db = await get_mongo_db()
    users_collection = db['users']
    try:
        user = await users_collection.find_one({'id_number': id_number})
        return user is not None
    except OperationFailure as e:
        logger.error(f"MongoDB operation failed during check_user_exists: {e}")
        return False
    except Exception as e:
        logger.error(f"Error checking user existence for '{id_number}': {e}")
        return False

async def get_user_by_id(id_number: str) -> Optional[Dict[str, Any]]:
    """Retrieves a user by their ID number."""
    db = await get_mongo_db()
    users_collection = db['users']
    try:
        user = await users_collection.find_one({'id_number': id_number})
        if user:
            # Convert list back to numpy array if needed for face recognition
            if 'face_embedding' in user and isinstance(user['face_embedding'], list):
                user['face_embedding'] = np.array(user['face_embedding'])
            return user
        return None
    except OperationFailure as e:
        logger.error(f"MongoDB operation failed during get_user_by_id: {e}")
        return None
    except Exception as e:
        logger.error(f"Error retrieving user '{id_number}': {e}")
        return None

async def get_all_users() -> List[Dict[str, Any]]:
    """Retrieves all registered users."""
    db = await get_mongo_db()
    users_collection = db['users']
    users = []
    try:
        # Use await with to_list() for asynchronous cursor iteration
        cursor = users_collection.find({})
        async for user in cursor: # Iterate asynchronously
            # Convert list back to numpy array if needed
            if 'face_embedding' in user and isinstance(user['face_embedding'], list):
                user['face_embedding'] = np.array(user['face_embedding'])
            users.append(user)
        logger.debug(f"Retrieved {len(users)} users from MongoDB.")
        return users
    except OperationFailure as e:
        logger.error(f"MongoDB operation failed during get_all_users: {e}")
        return []
    except Exception as e:
        logger.error(f"Error retrieving all users: {e}")
        return []

async def update_user_status(id_number: str, on_site: bool) -> bool:
    """Updates a user's on_site status."""
    db = await get_mongo_db()
    users_collection = db['users']
    try:
        result = await users_collection.update_one(
            {'id_number': id_number},
            {'$set': {'on_site': on_site}}
        )
        if result.modified_count > 0:
            logger.info(f"User '{id_number}' on_site status updated to {on_site}.")
            return True
        logger.warning(f"User '{id_number}' status not updated (user not found or status already set).")
        return False
    except OperationFailure as e:
        logger.error(f"MongoDB operation failed during update_user_status: {e}")
        return False
    except Exception as e:
        logger.error(f"Error updating user status for '{id_number}': {e}")
        return False

# --- Access Log Functions ---

async def log_access_event(user_id: str, user_name: str, action: str, method: str, confidence: Optional[float] = None) -> bool:
    """Logs an access event to MongoDB."""
    db = await get_mongo_db()
    access_logs_collection = db['access_logs']
    try:
        log_entry = {
            'user_id': user_id,
            'user_name': user_name,
            'action': action, # 'entry' or 'exit'
            'method': method, # 'face_recognition' or 'manual'
            'timestamp': datetime.utcnow(), # Store as UTC datetime object
            'confidence': confidence
        }
        await access_logs_collection.insert_one(log_entry)
        logger.info(f"Access event logged: User '{user_id}' ({user_name}) - {action} via {method}")
        return True
    except OperationFailure as e:
        logger.error(f"MongoDB operation failed during log_access_event: {e}")
        return False
    except Exception as e:
        logger.error(f"Error logging access event for user '{user_id}': {e}")
        return False

async def get_access_logs(user_id: Optional[str] = None, action: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """Retrieves access logs, optionally filtered by user_id or action."""
    db = await get_mongo_db()
    access_logs_collection = db['access_logs']
    query_filter = {}
    if user_id:
        query_filter['user_id'] = user_id
    if action:
        query_filter['action'] = action

    logs = []
    try:
        # Use await with to_list() for asynchronous cursor iteration
        cursor = access_logs_collection.find(query_filter).sort('timestamp', -1).limit(limit)
        async for log in cursor: # Iterate asynchronously
            # Convert ObjectId to string for JSON serialization if needed later
            if '_id' in log:
                log['_id'] = str(log['_id'])
            logs.append(log)
        logger.debug(f"Retrieved {len(logs)} access logs from MongoDB.")
        return logs
    except OperationFailure as e:
        logger.error(f"MongoDB operation failed during get_access_logs: {e}")
        return []
    except Exception as e:
        logger.error(f"Error retrieving access logs: {e}")
        return []

# --- Dashboard Specific Functions (Adapted for MongoDB) ---

async def get_dashboard_activities(limit: int = 50) -> List[Dict[str, Any]]:
    """Get all activities (recent logs) for the dashboard."""
    db = await get_mongo_db()
    access_logs_collection = db['access_logs']
    users_collection = db['users']
    activities = []
    try:
        # Fetch recent logs asynchronously
        recent_logs = await access_logs_collection.find().sort('timestamp', -1).limit(limit).to_list(length=None)

        # Get user names for the logs (can optimize with aggregation if performance is an issue)
        user_ids = [log['user_id'] for log in recent_logs]
        # Fetch users asynchronously
        users_cursor = users_collection.find({'id_number': {'$in': user_ids}})
        users_map = {user['id_number']: user['name'] async for user in users_cursor}

        for log in recent_logs:
            activity = {
                "id": str(log['_id']), # Convert ObjectId to string
                "user_id": log['user_id'],
                "userName": users_map.get(log['user_id'], 'Unknown User'),
                "action": log['action'],
                "timestamp": log['timestamp'].isoformat() if isinstance(log['timestamp'], datetime) else log['timestamp'],
                "method": log['method'],
                "confidence": log.get('confidence')
            }
            activities.append(activity)
        logger.debug(f"Fetched {len(activities)} dashboard activities.")
        return activities
    except OperationFailure as e:
        logger.error(f"MongoDB operation failed during get_dashboard_activities: {e}")
        return []
    except Exception as e:
        logger.error(f"Error fetching dashboard activities: {e}")
        return []

async def get_personnel_breakdown() -> Dict[str, int]:
    """Get personnel breakdown by role for currently on-site users."""
    db = await get_mongo_db()
    users_collection = db['users']
    breakdown = {role: 0 for role in VALID_ROLES} # Use the parsed VALID_ROLES
    try:
        pipeline = [
            {'$match': {'on_site': True}},
            {'$group': {'_id': '$role', 'count': {'$sum': 1}}}
        ]
        results = await users_collection.aggregate(pipeline).to_list(length=None) # Await aggregation results
        for res in results:
            if res['_id'] in breakdown:
                breakdown[res['_id']] = res['count']
        logger.debug(f"Personnel breakdown: {breakdown}")
        return breakdown
    except OperationFailure as e:
        logger.error(f"MongoDB operation failed during get_personnel_breakdown: {e}")
        return {role: 0 for role in VALID_ROLES}
    except Exception as e:
        logger.error(f"Error fetching personnel breakdown: {e}")
        return {role: 0 for role in VALID_ROLES}

async def get_total_on_site() -> int:
    """Get total personnel currently on site."""
    db = await get_mongo_db()
    users_collection = db['users']
    try:
        count = await users_collection.count_documents({'on_site': True}) # Await count_documents
        logger.debug(f"Total on site: {count}")
        return count
    except OperationFailure as e:
        logger.error(f"MongoDB operation failed during get_total_on_site: {e}")
        return 0
    except Exception as e:
        logger.error(f"Error fetching total on site: {e}")
        return 0

async def get_today_stats() -> Dict[str, Any]:
    """Get today's activity statistics."""
    db = await get_mongo_db()
    access_logs_collection = db['access_logs']

    start_of_day = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day = start_of_day + timedelta(days=1) - timedelta(microseconds=1)

    try:
        entries_count = await access_logs_collection.count_documents({ # Await count_documents
            'action': 'entry',
            'timestamp': {'$gte': start_of_day, '$lte': end_of_day}
        })
        exits_count = await access_logs_collection.count_documents({ # Await count_documents
            'action': 'exit',
            'timestamp': {'$gte': start_of_day, '$lte': end_of_day}
        })
        total_movements = entries_count + exits_count

        # Find peak hour
        pipeline = [
            {'$match': {'timestamp': {'$gte': start_of_day, '$lte': end_of_day}}},
            {'$group': {
                '_id': {'$hour': '$timestamp'},
                'count': {'$sum': 1}
            }},
            {'$sort': {'count': -1}},
            {'$limit': 1}
        ]
        peak_hour_result = await access_logs_collection.aggregate(pipeline).to_list(length=1) # Await aggregation

        peak_hour_str = 'N/A'
        if peak_hour_result:
            hour = peak_hour_result[0]['_id']
            peak_hour_str = f"{hour:02d}:00-{(hour + 1) % 24:02d}:00"

        stats = {
            "totalEntries": total_movements,
            "entries": entries_count,
            "exits": exits_count,
            "peakHour": peak_hour_str,
        }
        logger.debug(f"Today's stats: {stats}")
        return stats
    except OperationFailure as e:
        logger.error(f"MongoDB operation failed during get_today_stats: {e}")
        return {"totalEntries": 0, "entries": 0, "exits": 0, "peakHour": "N/A"}
    except Exception as e:
        logger.error(f"Error fetching today's stats: {e}")
        return {"totalEntries": 0, "entries": 0, "exits": 0, "peakHour": "N/A"}
