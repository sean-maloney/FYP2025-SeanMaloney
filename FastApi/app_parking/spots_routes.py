from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from motor.motor_asyncio import AsyncIOMotorDatabase
from .db import get_db

router = APIRouter(prefix="/api/cameras", tags=["spots"])



@router.post("/{camera_id}/spots")
async def save_spots_config(
    camera_id: str,
    payload: dict,
    publish: bool = False,
    db: AsyncIOMotorDatabase = Depends(get_db)
):
    if payload.get("camera_id") != camera_id:
        raise HTTPException(status_code=400, detail="Camera ID in URL and payload do not match.")
    
    latest = await db.spot_config.find_one({"camera_id": camera_id}, sort=[("version", -1)])
    next_version = (latest["version"] + 1) if latest else int(latest["version"]) + 1

    status = "published" if publish else "draft"

    if publish:
        await db.spot_config.update_many(
            {"camera_id": camera_id, "status": "published"},
            {"$set": {"status": "draft"}}
        )
    
    doc = {
        **payload,
        "camera_id": camera_id,
        "version": next_version,
        "status": status,
        "created_at": datetime.utcnow()
    }

    await db.spot_config.insert_one(doc)
    return {"camera_id": camera_id, "version": next_version, "status": status}



@router.get("/{camera_id}/spots/published")
async def get_published_spots_config(
    camera_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    doc = await db.spot_config.find_one(
        {"camera_id": camera_id, "status": "published"},
        sort=[("version", -1)]
    )
    if not doc:
        raise HTTPException(status_code=404, detail="No published spot config for this camera.")
    doc["_id"] = str(doc["_id"])
    return doc



@router.get("/{camera_id}/spots/history")
async def get_spots_history(
    camera_id:str,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    cursor = db.spot_config.find({"camera_id": camera_id}).sort("version", -1).limit(25)
    items = []
    async for d in cursor:
        d["_id"] = str(d["_id"])
        items.append(d)
    return items