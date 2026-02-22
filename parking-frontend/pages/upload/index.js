import { useState } from "react";
import { useRouter } from "next/router";

export default function Home() {
  const router = useRouter();

  const [file, setFile] = useState(null);
  const [cameraId, setCameraId] = useState("cam1");

  const [jobId, setJobId] = useState("");
  const [status, setStatus] = useState("");
  const [snapshotUrl, setSnapshotUrl] = useState("");

  const uploadVideo = async () => {
    if (!file) {
      setStatus("Select a video first");
      return;
    }

    setStatus("Uploading...");
    setSnapshotUrl("");
    setJobId("");

    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("camera_id", cameraId);

      const res = await fetch("http://127.0.0.1:8000/api/videos", {
        method: "POST",
        body: fd,
      });

      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || `HTTP ${res.status}`);
      }

      const data = await res.json();
      const newJobId = data.job_id;

      setJobId(newJobId);
      setStatus("Uploaded. Snapshot ready.");

      const snap = `http://127.0.0.1:8000/api/jobs/${newJobId}/snapshot?ts=${Date.now()}`;
      setSnapshotUrl(snap);
    } catch (err) {
      setStatus(`Upload failed: ${err.message}`);
    }
  };

  const goDraw = () => {
    router.push(`/spot-calibration?job_id=${jobId}&camera_id=${cameraId}`);
  };

  return (
    <div style={{ padding: 20, fontFamily: "Arial" }}>
      <h2>Upload Video</h2>

      <div style={{ marginBottom: 10 }}>
        <label style={{ marginRight: 8 }}>Camera ID:</label>
        <input value={cameraId} onChange={(e) => setCameraId(e.target.value)} />
      </div>

      <div style={{ marginBottom: 10 }}>
        <input
          type="file"
          accept="video/*"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
        />
      </div>

      <button onClick={uploadVideo}>Upload</button>

      <div style={{ marginTop: 12 }}>{status}</div>

      {jobId && (
        <div style={{ marginTop: 12 }}>
          <div>Job ID: {jobId}</div>
        </div>
      )}

      {snapshotUrl && (
        <div style={{ marginTop: 16 }}>
          <div style={{ marginBottom: 8 }}>Snapshot:</div>
          <img
            src={snapshotUrl}
            alt="snapshot"
            style={{ width: 960, maxWidth: "100%", border: "2px solid #444" }}
          />

          <div style={{ marginTop: 12 }}>
            <button onClick={goDraw}>Draw parking spots</button>
          </div>
        </div>
      )}
    </div>
  );
}
