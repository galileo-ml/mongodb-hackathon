"""Seed MongoDB with initial person data."""

import logging
import sys

from database import (
    create_person,
    delete_all_people,
    get_database,
    list_all_people,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("seed_mongodb")

# Initial person data - same as the original MOCK_PERSON_DATA
SEED_DATA = [
    {
        "person_id": "person_001",
        "name": "Sarah",
        "relationship": "Your daughter",
        "aggregated_context": "Works in corporate, recently got a promotion. Has children (your grandchildren) who visit occasionally. Cares about your wellbeing and checks in regularly.",
        "cached_description": "Last spoke 3 days ago about her promotion and the grandchildren visiting"
    },
    {
        "person_id": "person_002",
        "name": "Michael",
        "relationship": "Your son",
        "aggregated_context": "Regularly brings groceries and helps with errands. Enjoys outdoor activities like camping. Fixed his car recently. Very attentive to your needs.",
        "cached_description": "Visited yesterday with groceries and talked about his camping trip"
    },
    {
        "person_id": "person_003",
        "name": "Robert",
        "relationship": "Your friend from book club",
        "aggregated_context": "Old college friend who shares your love of mystery novels. Member of the same book club. Enjoys reminiscing about college memories and discussing books.",
        "cached_description": "Last week you discussed the mystery novel and college memories"
    }
]


def seed_database(reset: bool = False):
    """
    Seed the MongoDB database with initial person data.

    Args:
        reset: If True, delete all existing data before seeding
    """
    try:
        # Test connection
        db = get_database()
        logger.info("Successfully connected to MongoDB Atlas")

        # Optional: Reset database
        if reset:
            logger.warning("Resetting database - deleting all existing people...")
            deleted_count = delete_all_people()
            logger.info(f"Deleted {deleted_count} existing records")

        # Check if data already exists
        existing_people = list_all_people()
        if existing_people and not reset:
            logger.info(f"Database already has {len(existing_people)} people. Skipping seed.")
            logger.info("Use --reset flag to clear and re-seed the database")
            return

        # Insert seed data
        logger.info(f"Seeding database with {len(SEED_DATA)} people...")

        for person_data in SEED_DATA:
            create_person(
                person_id=person_data["person_id"],
                name=person_data["name"],
                relationship=person_data["relationship"],
                aggregated_context=person_data["aggregated_context"],
                cached_description=person_data["cached_description"]
            )

        # Verify
        all_people = list_all_people()
        logger.info(f"\n{'='*60}")
        logger.info("Database seeded successfully!")
        logger.info(f"{'='*60}")
        logger.info(f"Total people in database: {len(all_people)}")

        for person in all_people:
            logger.info(f"\n  - {person['name']} ({person['person_id']})")
            logger.info(f"    Relationship: {person['relationship']}")
            logger.info(f"    Description: {person['cached_description']}")

        logger.info(f"{'='*60}\n")

    except Exception as e:
        logger.error(f"Error seeding database: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Check for --reset flag
    reset = "--reset" in sys.argv

    if reset:
        logger.warning("Reset flag detected - will delete all existing data!")

    seed_database(reset=reset)
