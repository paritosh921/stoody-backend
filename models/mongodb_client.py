"""
MongoDB Client for SkillBot Backend
Handles database connection and provides database instance
"""

import logging
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ConfigurationError
from config import MONGODB_URI, MONGODB_DB_NAME

logger = logging.getLogger(__name__)

class MongoDBClient:
    """MongoDB client wrapper for SkillBot database operations"""

    def __init__(self):
        self.client = None
        self.db = None
        self._connect()

    def _connect(self):
        """Establish connection to MongoDB"""
        try:
            if not MONGODB_URI:
                raise ConfigurationError("MONGODB_URI is not configured")

            # Create MongoDB client
            self.client = MongoClient(MONGODB_URI)

            # Test the connection
            self.client.admin.command('ping')

            # Get database instance
            self.db = self.client[MONGODB_DB_NAME]

            logger.info(f"Successfully connected to MongoDB database: {MONGODB_DB_NAME}")

        except ConnectionFailure as e:
            logger.warning(f"Failed to connect to MongoDB: {str(e)}")
            logger.warning("MongoDB connection failed - running in offline mode")
            self.client = None
            self.db = None
        except ConfigurationError as e:
            logger.warning(f"MongoDB configuration error: {str(e)}")
            logger.warning("MongoDB not configured - running in offline mode")
            self.client = None
            self.db = None
        except Exception as e:
            logger.error(f"Unexpected error connecting to MongoDB: {str(e)}")
            raise

    def get_database(self):
        """Get database instance"""
        if self.db is None:
            self._connect()
        return self.db

    def get_collection(self, collection_name):
        """Get collection instance"""
        return self.get_database()[collection_name]

    def close_connection(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")

    def health_check(self):
        """Check if MongoDB connection is healthy"""
        try:
            if self.client is None:
                return False

            # Ping the database
            self.client.admin.command('ping')
            return True

        except Exception as e:
            logger.error(f"MongoDB health check failed: {str(e)}")
            return False

# Global MongoDB client instance
mongo_client = MongoDBClient()

def get_db():
    """Get database instance - convenience function"""
    return mongo_client.get_database()

def get_collection(collection_name):
    """Get collection instance - convenience function"""
    return mongo_client.get_collection(collection_name)