from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch


def test_upload_video_no_file(client):
    r = client.post("/api/videos", data={"camera_id": "cam1"})
    assert r.status_code == 422


def test_upload_video_wrong_type(client):
    r = client.post("/api/videos", files={"file": ("test.txt", BytesIO(b"data"), "text/plain")}, data={"camera_id": "cam1"})
    assert r.status_code == 400
    assert "video" in r.json()["detail"]


def test_upload_video_success(client):
    mock_db = MagicMock()
    mock_db.jobs.insert_one = AsyncMock()
    with patch("app_parking.main.get_db", return_value=mock_db):
        r = client.post("/api/videos", files={"file": ("test.mp4", BytesIO(b"fake"), "video/mp4")}, data={"camera_id": "cam1"})
    assert r.status_code == 201
    assert "job_id" in r.json()
    assert r.json()["camera_id"] == "cam1"


def test_save_grid_missing_camera_id(client):
    r = client.post("/api/pathfinder/grid/save", json={"rows": 2, "cols": 2, "grid": [[0, 0], [0, 0]], "start": [0, 0]})
    assert r.status_code == 400

def test_save_grid_row_count_mismatch(client, tmp_path):
    with patch("app_parking.routes.pathfinder.GRID_CONFIG_DIR", tmp_path):
        r = client.post("/api/pathfinder/grid/save", json={
            "camera_id": "cam1",
            "rows": 3,
            "cols": 2,
            "grid": [[0, 0], [0, 0]],
            "start": [0, 0],
            "parking_spaces": [],
        })
    assert r.status_code == 400
    assert "row" in r.json()["detail"]


def test_save_grid_col_count_mismatch(client, tmp_path):
    with patch("app_parking.routes.pathfinder.GRID_CONFIG_DIR", tmp_path):
        r = client.post("/api/pathfinder/grid/save", json={
            "camera_id": "cam1",
            "rows": 2,
            "cols": 3,
            "grid": [[0, 0], [0, 0]],
            "start": [0, 0],
            "parking_spaces": [],
        })
    assert r.status_code == 400
    assert "col" in r.json()["detail"]


def test_save_grid_zero_rows(client, tmp_path):
    with patch("app_parking.routes.pathfinder.GRID_CONFIG_DIR", tmp_path):
        r = client.post("/api/pathfinder/grid/save", json={
            "camera_id": "cam1",
            "rows": 0,
            "cols": 2,
            "grid": [],
            "start": [0, 0],
            "parking_spaces": [],
        })
    assert r.status_code == 400


def test_save_grid_start_outside_grid(client, tmp_path):
    with patch("app_parking.routes.pathfinder.GRID_CONFIG_DIR", tmp_path):
        r = client.post("/api/pathfinder/grid/save", json={
            "camera_id": "cam1",
            "rows": 2,
            "cols": 2,
            "grid": [[0, 0], [0, 0]],
            "start": [5, 5],
            "parking_spaces": [],
        })
    assert r.status_code == 400
    assert "start" in r.json()["detail"]


def test_save_grid_start_negative(client, tmp_path):
    with patch("app_parking.routes.pathfinder.GRID_CONFIG_DIR", tmp_path):
        r = client.post("/api/pathfinder/grid/save", json={
            "camera_id": "cam1",
            "rows": 2,
            "cols": 2,
            "grid": [[0, 0], [0, 0]],
            "start": [-1, 0],
            "parking_spaces": [],
        })
    assert r.status_code == 400
    assert "start" in r.json()["detail"]


def test_save_grid_parking_space_outside_grid(client, tmp_path):
    with patch("app_parking.routes.pathfinder.GRID_CONFIG_DIR", tmp_path):
        r = client.post("/api/pathfinder/grid/save", json={
            "camera_id": "cam1",
            "rows": 2,
            "cols": 2,
            "grid": [[0, 0], [0, 0]],
            "start": [0, 0],
            "parking_spaces": [[5, 5]],
        })
    assert r.status_code == 400
    assert "parking" in r.json()["detail"]


def test_save_grid_parking_space_negative(client, tmp_path):
    with patch("app_parking.routes.pathfinder.GRID_CONFIG_DIR", tmp_path):
        r = client.post("/api/pathfinder/grid/save", json={
            "camera_id": "cam1",
            "rows": 2,
            "cols": 2,
            "grid": [[0, 0], [0, 0]],
            "start": [0, 0],
            "parking_spaces": [[-1, 0]],
        })
    assert r.status_code == 400
    assert "parking" in r.json()["detail"]


def test_astar_start_equals_goal():
    from app_parking.services.astar import run_astar_process
    grid = [[0] * 5 for _ in range(5)]
    result = run_astar_process("test", 5, 5, [2, 2], [2, 2], grid)
    assert result["success"] is False


def test_save_spots_camera_id_mismatch(client):
    r = client.post("/api/cameras/cam1/spots", json={
        "camera_id": "cam2",
        "spots": [],
    })
    assert r.status_code == 400
    assert "match" in r.json()["detail"].lower()


def test_job_snapshot_not_found(client):
    r = client.get("/api/jobs/nonexistent-job-id/snapshot")
    assert r.status_code == 404
    assert "not found" in r.json()["detail"].lower()


def test_save_grid_success(client, tmp_path):
    with patch("app_parking.routes.pathfinder.GRID_CONFIG_DIR", tmp_path):
        r = client.post("/api/pathfinder/grid/save", json={
            "camera_id": "testcam", "rows": 2, "cols": 2,
            "grid": [[0, 0], [0, 0]], "start": [0, 0], "parking_spaces": [[1, 1]],
        })
    assert r.status_code == 200
    assert r.json()["camera_id"] == "testcam"


def test_load_grid_not_found(client, tmp_path):
    with patch("app_parking.routes.pathfinder.GRID_CONFIG_DIR", tmp_path):
        r = client.get("/api/pathfinder/grid/nonexistent")
    assert r.status_code == 404


def test_load_grid_success(client, tmp_path):
    import json
    grid_data = {"camera_id": "cam1", "rows": 2, "cols": 2, "grid": [[0, 0], [0, 0]], "start": [0, 0], "parking_spaces": []}
    (tmp_path / "cam1.json").write_text(json.dumps(grid_data))
    with patch("app_parking.routes.pathfinder.GRID_CONFIG_DIR", tmp_path):
        r = client.get("/api/pathfinder/grid/cam1")
    assert r.status_code == 200
    assert r.json()["camera_id"] == "cam1"


def test_astar_direct_path():
    from app_parking.services.astar import run_astar_process
    grid = [[0] * 5 for _ in range(5)]
    result = run_astar_process("test", 5, 5, [0, 0], [4, 4], grid)
    assert result["success"] is True
    assert result["path"][0] == [0, 0]
    assert result["path"][-1] == [4, 4]


def test_astar_blocked_goal():
    from app_parking.services.astar import run_astar_process
    grid = [[0] * 5 for _ in range(5)]
    grid[4][4] = 1
    result = run_astar_process("test", 5, 5, [0, 0], [4, 4], grid)
    assert result["success"] is False


def test_astar_no_path():
    from app_parking.services.astar import run_astar_process
    grid = [[0, 1, 0], [0, 1, 0], [0, 1, 0]]
    result = run_astar_process("test", 3, 3, [0, 0], [0, 2], grid)
    assert result["success"] is False


def test_is_spot_available_status_string():
    from app_parking.services.astar import is_spot_available
    assert is_spot_available({"status": "available"}) is True
    assert is_spot_available({"status": "occupied"}) is False


def test_is_spot_available_bool_fields():
    from app_parking.services.astar import is_spot_available
    assert is_spot_available({"occupied": False}) is True
    assert is_spot_available({"available": True}) is True


def test_point_in_polygon_inside():
    from app_parking.services.astar import point_in_polygon
    polygon = [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}, {"x": 1.0, "y": 1.0}, {"x": 0.0, "y": 1.0}]
    assert point_in_polygon(0.5, 0.5, polygon) is True


def test_point_in_polygon_outside():
    from app_parking.services.astar import point_in_polygon
    polygon = [{"x": 0.0, "y": 0.0}, {"x": 0.5, "y": 0.0}, {"x": 0.5, "y": 0.5}, {"x": 0.0, "y": 0.5}]
    assert point_in_polygon(0.9, 0.9, polygon) is False
