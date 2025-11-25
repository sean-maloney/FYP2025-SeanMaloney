# tests/test_parking.py

from io import BytesIO


def test_status_endpoint(client):
  r = client.get("/api/status")
  assert r.status_code == 200
  data = r.json()
  assert "model_loaded" in data
  assert "device" in data


def test_upload_frame_and_parking_status(client):
  # send a fake image (we only care about content_type)
  fake_image = BytesIO(b"fake image data")
  files = {"file": ("test.png", fake_image, "image/png")}

  r = client.post("/api/frames", files=files)
  assert r.status_code == 201

  data = r.json()
  assert "total_spaces" in data
  assert "free_spaces" in data
  assert "occupied_spaces" in data
  assert isinstance(data["spaces"], list)
  assert len(data["spaces"]) == data["total_spaces"]

  # now /api/parking-status should return a summary
  r2 = client.get("/api/parking-status")
  assert r2.status_code == 200
  summary = r2.json()
  assert "video_id" in summary
  assert "total_detections" in summary
  assert "status" in summary


def test_upload_video_background(client):
  fake_video = BytesIO(b"fake video data")
  files = {"file": ("video.mp4", fake_video, "video/mp4")}

  r = client.post("/api/videos", files=files)
  assert r.status_code == 200
  data = r.json()
  assert data["status"] == "processing"
  assert "video_id" in data
