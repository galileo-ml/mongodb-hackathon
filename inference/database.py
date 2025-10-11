"""MongoDB database operations for person data."""

import logging
import os
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("database")

# MongoDB configuration
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "dementia_care_db")

# Global MongoDB client and database
_client: Optional[MongoClient] = None
_db: Optional[Database] = None


def get_database() -> Database:
    """Get MongoDB database instance (singleton pattern)."""
    global _client, _db

    if _db is None:
        if not MONGODB_URI:
            raise ValueError("MONGODB_URI environment variable not set")

        logger.info("Connecting to MongoDB Atlas...")
        _client = MongoClient(MONGODB_URI)
        _db = _client[MONGODB_DATABASE]

        # Test connection
        _client.admin.command('ping')
        logger.info(f"Connected to MongoDB database: {MONGODB_DATABASE}")

    return _db


def get_people_collection() -> Collection:
    """Get the 'people' collection."""
    db = get_database()
    return db["people"]


def get_person_by_id(person_id: str) -> Optional[dict]:
    """
    Retrieve a person document by person_id.

    Args:
        person_id: The person identifier

    Returns:
        Person document or None if not found
    """
    collection = get_people_collection()
    person_doc = collection.find_one({"person_id": person_id})

    if person_doc:
        logger.info(f"Found person: {person_doc.get('name')} ({person_id})")
    else:
        logger.warning(f"Person not found: {person_id}")

    return person_doc


def create_person(
    person_id: str,
    name: str,
    relationship: str,
    aggregated_context: str = "",
    cached_description: str = "No previous interactions"
) -> dict:
    """
    Create a new person document.

    Args:
        person_id: Unique person identifier
        name: Person's name
        relationship: Relationship to patient
        aggregated_context: Running summary of conversations
        cached_description: One-line description for AR display

    Returns:
        Created person document
    """
    collection = get_people_collection()

    person_doc = {
        "person_id": person_id,
        "name": name,
        "relationship": relationship,
        "aggregated_context": aggregated_context,
        "cached_description": cached_description,
        "last_updated": datetime.utcnow()
    }

    result = collection.insert_one(person_doc)
    logger.info(f"Created person: {name} ({person_id})")

    return person_doc


def update_person_context(
    person_id: str,
    aggregated_context: str,
    cached_description: str
) -> bool:
    """
    Update a person's aggregated context and cached description.

    Args:
        person_id: Person identifier
        aggregated_context: Updated conversation summary
        cached_description: New one-line description

    Returns:
        True if updated, False if person not found
    """
    collection = get_people_collection()

    result = collection.update_one(
        {"person_id": person_id},
        {
            "$set": {
                "aggregated_context": aggregated_context,
                "cached_description": cached_description,
                "last_updated": datetime.utcnow()
            }
        }
    )

    if result.matched_count > 0:
        logger.info(f"Updated context for person: {person_id}")
        return True
    else:
        logger.warning(f"Person not found for update: {person_id}")
        return False


def list_all_people() -> list[dict]:
    """
    List all people in the database.

    Returns:
        List of person documents
    """
    collection = get_people_collection()
    people = list(collection.find())
    logger.info(f"Found {len(people)} people in database")
    return people


def delete_all_people() -> int:
    """
    Delete all people from the database (for testing/reset).

    Returns:
        Number of documents deleted
    """
    collection = get_people_collection()
    result = collection.delete_many({})
    logger.info(f"Deleted {result.deleted_count} people from database")
    return result.deleted_count


def close_connection():
    """Close MongoDB connection."""
    global _client, _db

    if _client:
        _client.close()
        _client = None
        _db = None
        logger.info("MongoDB connection closed")
