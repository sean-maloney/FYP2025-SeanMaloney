import os
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "fyp_parking")

_client: Optional[AsyncIOMotorClient] = None

async def connect_mongo() -> None:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(MONGO_URI)

        db = _client[MONGO_DB]
        await db.jobs.create_index("status")
        await db.jobs.create_index("created_at")
        await db.spot_config.create_index([("camera_id", 1), ("status", 1)])
        await db.spot_config.create_index([("camera_id", 1), ("version", -1)])

async def close_mongo() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None

def get_db() -> AsyncIOMotorDatabase:
    if _client is None:
        raise Exception("MongoDB client is not connected. Call connect_mongo() first.")
    return _client[MONGO_DB]