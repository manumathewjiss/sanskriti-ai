"""
MongoDB Atlas connection and database operations module.
Provides async database client and helper functions for CRUD operations.
"""

import motor.motor_asyncio
from pymongo import errors
from typing import Optional, List, Dict, Any
from bson import ObjectId

from config import Config
from logger import get_logger

logger = get_logger(__name__)


class MongoDB:
    """MongoDB Atlas client wrapper with connection management."""
    
    def __init__(self):
        """Initialize MongoDB client (connection is lazy)."""
        self.client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
        self.db = None
        
    async def connect(self):
        """
        Connect to MongoDB Atlas.
        
        Raises:
            Exception: If connection fails
        """
        try:
            logger.info("Connecting to MongoDB Atlas...")
            self.client = motor.motor_asyncio.AsyncIOMotorClient(
                Config.MONGODB_URL,
                serverSelectionTimeoutMS=5000
            )
            
            # Test connection
            await self.client.admin.command('ping')
            
            self.db = self.client[Config.MONGODB_DB_NAME]
            logger.info(f"✓ Connected to MongoDB database: {Config.MONGODB_DB_NAME}")
            
        except errors.ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error connecting to MongoDB: {e}")
            raise
    
    async def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
    
    def get_collection(self, collection_name: str):
        """
        Get a MongoDB collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            AsyncIOMotorCollection instance
        """
        if self.db is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self.db[collection_name]
    
    async def insert_one(self, collection_name: str, document: Dict[str, Any]) -> ObjectId:
        """
        Insert a single document.
        
        Args:
            collection_name: Collection to insert into
            document: Document to insert
            
        Returns:
            ObjectId of inserted document
        """
        collection = self.get_collection(collection_name)
        result = await collection.insert_one(document)
        return result.inserted_id
    
    async def insert_many(
        self, 
        collection_name: str, 
        documents: List[Dict[str, Any]]
    ) -> List[ObjectId]:
        """
        Insert multiple documents.
        
        Args:
            collection_name: Collection to insert into
            documents: List of documents to insert
            
        Returns:
            List of ObjectIds of inserted documents
        """
        if not documents:
            return []
        
        collection = self.get_collection(collection_name)
        result = await collection.insert_many(documents)
        return result.inserted_ids
    
    async def find_one(
        self, 
        collection_name: str, 
        query: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Find a single document.
        
        Args:
            collection_name: Collection to search
            query: Query filter
            
        Returns:
            Document or None if not found
        """
        collection = self.get_collection(collection_name)
        return await collection.find_one(query)
    
    async def find_many(
        self, 
        collection_name: str, 
        query: Dict[str, Any],
        limit: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Find multiple documents.
        
        Args:
            collection_name: Collection to search
            query: Query filter
            limit: Maximum number of documents to return (0 = no limit)
            
        Returns:
            List of documents
        """
        collection = self.get_collection(collection_name)
        cursor = collection.find(query)
        
        if limit > 0:
            cursor = cursor.limit(limit)
        
        return await cursor.to_list(length=None)
    
    async def delete_one(
        self, 
        collection_name: str, 
        query: Dict[str, Any]
    ) -> int:
        """
        Delete a single document.
        
        Args:
            collection_name: Collection to delete from
            query: Query filter
            
        Returns:
            Number of deleted documents
        """
        collection = self.get_collection(collection_name)
        result = await collection.delete_one(query)
        return result.deleted_count
    
    async def delete_many(
        self, 
        collection_name: str, 
        query: Dict[str, Any]
    ) -> int:
        """
        Delete multiple documents.
        
        Args:
            collection_name: Collection to delete from
            query: Query filter
            
        Returns:
            Number of deleted documents
        """
        collection = self.get_collection(collection_name)
        result = await collection.delete_many(query)
        return result.deleted_count
    
    async def count_documents(
        self, 
        collection_name: str, 
        query: Dict[str, Any] = None
    ) -> int:
        """
        Count documents matching query.
        
        Args:
            collection_name: Collection to count from
            query: Query filter (None = count all)
            
        Returns:
            Number of documents
        """
        collection = self.get_collection(collection_name)
        if query is None:
            query = {}
        return await collection.count_documents(query)


# Global database instance
db_instance = MongoDB()


async def get_database() -> MongoDB:
    """
    Get the global database instance.
    Ensures connection is established.
    
    Returns:
        MongoDB instance
    """
    if db_instance.client is None:
        await db_instance.connect()
    return db_instance
