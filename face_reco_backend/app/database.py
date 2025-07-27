import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get MongoDB connection details from environment variables
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB_NAME")
COLLECTION_NAME = "registered_faces" # You can keep this hardcoded or also make it an environment variable

if not MONGO_URI:
    raise ValueError("MONGO_URI environment variable not set.")
if not DB_NAME:
    raise ValueError("MONGO_DB_NAME environment variable not set.")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

def get_db_collection():
    return collection