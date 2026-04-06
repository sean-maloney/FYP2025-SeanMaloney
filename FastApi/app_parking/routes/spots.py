from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from ..core.db import get_db

router = APIRouter(prefix="/api/cameras", tags=["Spots"])


@router.post("/{camera_id}/spots")
async def save_spots_config(
    camera_id: str,
    payload: dict,
    publish: bool = False,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    if payload.get("camera_id") != camera_id:
        raise HTTPException(status_code=400, detail="Camera ID in URL and payload do not match.")

    spots = payload.get("spots", [])
    for s in spots:
        s.setdefault("type", "parking")
        s.setdefault("description", "")
    payload["spots"] = spots

    latest = await db.spot_configs.find_one({"camera_id": camera_id}, sort=[("version", -1)])
    next_version = 1 if not latest else int(latest["version"]) + 1
    status = "published" if publish else "draft"

    if publish:
        await db.spot_configs.update_many(
            {"camera_id": camera_id, "status": "published"},
            {"$set": {"status": "draft"}},
        )

    await db.spot_configs.insert_one({
        **payload,
        "camera_id": camera_id,
        "version": next_version,
        "status": status,
        "created_at": datetime.utcnow(),
    })
    return {"camera_id": camera_id, "version": next_version, "status": status}


@router.get("/{camera_id}/spots/published")
async def get_published_spots_config(
    camera_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    doc = await db.spot_configs.find_one(
        {"camera_id": camera_id, "status": "published"},
        sort=[("version", -1)],
    )
    if not doc:
        raise HTTPException(status_code=404, detail="No published spot config for this camera.")
    doc["_id"] = str(doc["_id"])
    return doc


@router.get("/{camera_id}/spots/history")
async def get_spots_history(
    camera_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    cursor = db.spot_configs.find({"camera_id": camera_id}).sort("version", -1).limit(25)
    items = []
    async for d in cursor:
        d["_id"] = str(d["_id"])
        items.append(d)
    return items
