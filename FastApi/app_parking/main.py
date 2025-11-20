from contextlib import asynccontextmanager
from pathlib import Path
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .config import UPLOAD_DIR, OUTPUT_DIR
from .yolo_service import load_yolo_model, run_yolo_on_video

# simple in-memory counter for jobs
job_counter = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    load_yolo_model()
    yield
    # shutdown (nothing yet)


app = FastAPI(
    title="Parking Detector API",
    description="Simple API: upload a video, run YOLO, then fetch processed video.",
    lifespan=lifespan,
)

# CORS similar style to your class app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


# 1) upload + process
@app.post("/api/videos", status_code=status.HTTP_201_CREATED)
async def upload_video(file: UploadFile = File(...)): #... to make sure a file is uploaded
    global job_counter
    job_counter += 1
    job_id = job_counter

    if not file.filename:
        raise HTTPException(status_code=400, detail="no file uploaded")

    if not file.content_type.startswith("video"): #checks if the file is a video
        raise HTTPException(status_code=400, detail="file must be a video")

    # save uploaded file as "<job_id>.mp4"
    upload_path = UPLOAD_DIR / f"{job_id}.mp4"

    with upload_path.open("wb") as buffer: #w=wrtie mode b=binary mode needed for videos and stuff
        shutil.copyfileobj(file.file, buffer) #take the video uploaded, copy it's raw bytes, store it locally on disk

    try:
        output_mp4 = run_yolo_on_video(upload_path, str(job_id))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"error running yolo: {e}")

    return {
        "job_id": job_id,
        "output_file": output_mp4.name,
        "message": f"upload complete, call GET /api/videos/{job_id} to fetch result",
    }


# 2) get processed video
@app.get("/api/videos/{job_id}", response_class=FileResponse) #not returning json, returning a file
async def get_video(job_id: int): #gets the video by the correlating id
    job_dir = OUTPUT_DIR / str(job_id) #will make a folder for every job, so it doesnt get confusing
    video_path = job_dir / "output_fixed.mp4" #creates the expected file path

    if not video_path.exists(): #does the video actually exists, name might be wrong
        mp4_list = list(job_dir.glob("*.mp4"))# so we actually look for any mp4 there
        #if no mp4 for whatever reason, we inform the user
        if not mp4_list:
            raise HTTPException(status_code=404, detail="processed video not found")
        video_path = mp4_list[0] #if there is one .mp4 use the most recent/first one as the output video.

    #sends us back the actual video
    return FileResponse(
        path=str(video_path), #where it is on disc
        media_type="video/mp4", #tell the browser it's a video
        filename=video_path.name,  #changes the name of the download file
    )


#add photo option

#add live stream (main one I want to use)