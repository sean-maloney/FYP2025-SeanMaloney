import { useMemo, useState } from "react";
import ParkingGridEditor from "../components/grid/ParkingGridEditor";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";

export default function GridSetupPage() {
  const [cameraId, setCameraId] = useState("cam1");
  const [videoFile, setVideoFile] = useState(null);
  const [imageUrl, setImageUrl] = useState("");
  const [sourceType, setSourceType] = useState("none");
  const [rows, setRows] = useState(12);
  const [cols, setCols] = useState(16);
  const [status, setStatus] = useState("Choose an image source to begin.");
  const [error, setError] = useState("");
  const [editorKey, setEditorKey] = useState(0);

  const resolvedImageUrl = useMemo(() => {
    if (!imageUrl) return "";
    return imageUrl.startsWith("http") ? imageUrl : `${API_BASE}${imageUrl}`;
  }, [imageUrl]);

  async function loadFromCamera() {
    setError("");
    setStatus("Loading latest camera snapshot...");

    try {
      const response = await fetch(`${API_BASE}/api/cameras/${cameraId}/capture`, {
        method: "POST",
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Could not load camera snapshot.");
      }

      setImageUrl(`/api/cameras/${cameraId}/image?t=${Date.now()}`);
      setSourceType("camera");
      setEditorKey((value) => value + 1);
      setStatus("Camera snapshot loaded into grid editor.");
    } catch (err) {
      setError(err.message || "Could not load camera snapshot.");
      setStatus("Camera load failed.");
    }
  }

  async function loadFromVideo() {
    if (!videoFile) {
      setError("Please choose a video file first.");
      return;
    }

    setError("");
    setStatus("Extracting a frame from uploaded video...");

    try {
      const formData = new FormData();
      formData.append("file", videoFile);
      formData.append("camera_id", cameraId);

      const response = await fetch(`${API_BASE}/api/pathfinder/grid-source/video`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Could not extract frame from video.");
      }

      setImageUrl(`${data.image_url}?t=${Date.now()}`);
      setSourceType("video");
      setEditorKey((value) => value + 1);
      setStatus("Video frame loaded into grid editor.");
    } catch (err) {
      setError(err.message || "Could not extract frame from video.");
      setStatus("Video frame extraction failed.");
    }
  }

  return (
    <div style={{ padding: "24px", maxWidth: "1400px", margin: "0 auto" }}>
      <h1 style={{ marginBottom: "8px" }}>Grid Setup</h1>
      <p style={{ marginTop: 0, color: "#4b5563" }}>
        Load the editor background from the latest camera snapshot or from an uploaded video frame.
      </p>

      <div
        style={{
          background: "#f8fafc",
          border: "1px solid #e5e7eb",
          borderRadius: 12,
          padding: 16,
          marginBottom: 20,
        }}
      >
        <div
          style={{
            display: "flex",
            gap: 12,
            flexWrap: "wrap",
            marginBottom: 12,
            alignItems: "center",
          }}
        >
          <input
            value={cameraId}
            onChange={(e) => setCameraId(e.target.value)}
            placeholder="camera id"
            style={{
              padding: "10px 12px",
              borderRadius: 8,
              border: "1px solid #d1d5db",
              minWidth: 200,
            }}
          />

          <label>
            Rows{" "}
            <input
              type="number"
              min="1"
              value={rows}
              onChange={(e) => setRows(Number(e.target.value))}
              style={{
                padding: "10px 12px",
                borderRadius: 8,
                border: "1px solid #d1d5db",
                width: 90,
              }}
            />
          </label>

          <label>
            Cols{" "}
            <input
              type="number"
              min="1"
              value={cols}
              onChange={(e) => setCols(Number(e.target.value))}
              style={{
                padding: "10px 12px",
                borderRadius: 8,
                border: "1px solid #d1d5db",
                width: 90,
              }}
            />
          </label>

          <button
            onClick={loadFromCamera}
            style={{
              padding: "10px 14px",
              borderRadius: 8,
              border: "1px solid #d1d5db",
              cursor: "pointer",
              background: "white",
            }}
          >
            Load from Camera
          </button>
        </div>

        <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "center" }}>
          <input
            type="file"
            accept="video/*"
            onChange={(e) => setVideoFile(e.target.files?.[0] || null)}
          />

          <button
            onClick={loadFromVideo}
            style={{
              padding: "10px 14px",
              borderRadius: 8,
              border: "1px solid #d1d5db",
              cursor: "pointer",
              background: "white",
            }}
          >
            Extract Frame from Video
          </button>
        </div>

        <div style={{ marginTop: 14, fontSize: 14 }}>
          <strong>Status:</strong> {status}
        </div>

        <div style={{ marginTop: 8, fontSize: 14 }}>
          <strong>Source:</strong> {sourceType}
        </div>

        {error ? (
          <div
            style={{
              marginTop: 12,
              padding: "10px 12px",
              border: "1px solid #ef4444",
              background: "#fef2f2",
              color: "#b91c1c",
              borderRadius: 8,
              fontSize: "14px",
              maxWidth: "760px",
            }}
          >
            <strong>Error:</strong> {error}
          </div>
        ) : null}
      </div>

      <ParkingGridEditor
        key={`${cameraId}-${rows}-${cols}-${editorKey}`}
        initialCameraId={cameraId}
        backgroundImageUrl={resolvedImageUrl}
        initialRows={rows}
        initialCols={cols}
      />
    </div>
  );
}