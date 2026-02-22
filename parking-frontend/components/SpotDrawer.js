import React, { useEffect, useRef, useState } from "react";
import { useRouter } from "next/router";

function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}

function toNormPoint(e, canvas) {
  const rect = canvas.getBoundingClientRect();
  const xPx = e.clientX - rect.left;
  const yPx = e.clientY - rect.top;
  return {
    x: clamp01(xPx / rect.width),
    y: clamp01(yPx / rect.height),
  };
}

export default function SpotDrawer({ jobId, cameraId }) {
  const router = useRouter();
  const canvasRef = useRef(null);

  const [bgImg, setBgImg] = useState(null);

  const [currentPoly, setCurrentPoly] = useState([]);
  const [spots, setSpots] = useState([]);

  const [spotId, setSpotId] = useState("A01");
  const [spotType, setSpotType] = useState("parking");
  const [spotDesc, setSpotDesc] = useState("");

  const [publish, setPublish] = useState(true);
  const [statusMsg, setStatusMsg] = useState("");
  const [inferenceDone, setInferenceDone] = useState(false);

  useEffect(() => {
    const url = `http://127.0.0.1:8000/api/jobs/${jobId}/snapshot?ts=${Date.now()}`;
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => setBgImg(img);
    img.onerror = () => setBgImg(null);
    img.src = url;
  }, [jobId]);

  const redraw = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);

    if (bgImg) {
      ctx.drawImage(bgImg, 0, 0, w, h);
    } else {
      ctx.fillStyle = "#222";
      ctx.fillRect(0, 0, w, h);
    }

    spots.forEach((s) => {
      const pts = s.polygon.map((p) => ({ x: p.x * w, y: p.y * h }));
      if (pts.length < 3) return;

      ctx.lineWidth = 2;
      ctx.strokeStyle = "yellow";
      ctx.fillStyle = "rgba(255,255,0,0.10)";

      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = "yellow";
      ctx.font = "16px Arial";
      ctx.fillText(`${s.id} (${s.type})`, pts[0].x + 6, pts[0].y - 6);
    });

    if (currentPoly.length > 0) {
      const pts = currentPoly.map((p) => ({ x: p.x * w, y: p.y * h }));

      ctx.lineWidth = 2;
      ctx.strokeStyle = "cyan";
      ctx.fillStyle = "rgba(0,255,255,0.10)";

      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
      if (pts.length >= 3) ctx.closePath();
      ctx.stroke();
      if (pts.length >= 3) ctx.fill();

      ctx.fillStyle = "cyan";
      pts.forEach((p) => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
        ctx.fill();
      });
    }
  };

  useEffect(() => {
    redraw();
  }, [bgImg, spots, currentPoly]);

  const onCanvasClick = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const p = toNormPoint(e, canvas);
    setCurrentPoly((prev) => [...prev, p]);
  };

  const undoPoint = () => setCurrentPoly((prev) => prev.slice(0, -1));
  const clearCurrent = () => setCurrentPoly([]);

  const addSpot = () => {
    if (currentPoly.length < 3) {
      setStatusMsg("Need at least 3 points");
      return;
    }

    const id = spotId.trim();
    if (!id) {
      setStatusMsg("Spot ID required");
      return;
    }

    setSpots((prev) => [
      ...prev,
      {
        id,
        type: spotType,
        description: spotDesc,
        polygon: currentPoly,
      },
    ]);

    setCurrentPoly([]);
    setSpotDesc("");

    setSpotId((prev) => {
      const m = prev.match(/^([A-Za-z]*)(\d+)$/);
      if (!m) return prev;
      const prefix = m[1];
      const num = String(parseInt(m[2], 10) + 1).padStart(m[2].length, "0");
      return `${prefix}${num}`;
    });

    setStatusMsg("");
  };

  const removeLastSpot = () => setSpots((prev) => prev.slice(0, -1));

  const saveSpots = async () => {
    const payload = {
      camera_id: cameraId,
      frame_size: { w: 1280, h: 720 },
      spots,
    };

    const url = `http://127.0.0.1:8000/api/cameras/${cameraId}/spots?publish=${publish ? "true" : "false"}`;
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const txt = await res.text();
      throw new Error(txt || `HTTP ${res.status}`);
    }

    return await res.json();
  };

  const runJob = async () => {
    const url = `http://127.0.0.1:8000/api/jobs/${jobId}/run`;
    const res = await fetch(url, { method: "POST" });

    if (!res.ok) {
      const txt = await res.text();
      throw new Error(txt || `HTTP ${res.status}`);
    }

    return await res.json();
  };

  const saveAndRun = async () => {
    setStatusMsg("Saving spots...");
    setInferenceDone(false);
    try {
      const saved = await saveSpots();
      setStatusMsg(`Saved (v${saved.version}). Running inference...`);
      const out = await runJob();
      setStatusMsg(`Done. Available=${out.available} Occupied=${out.occupied}`);
      setInferenceDone(true);
    } catch (err) {
      setStatusMsg(`Failed: ${err.message}`);
    }
  };

  const goView = () => {
    router.push(`/view/${jobId}`);
  };

  return (
    <div style={{ padding: 20, fontFamily: "Arial" }}>
      <h2>Spot Calibration</h2>

      <div style={{ marginBottom: 10 }}>
        <div>Job ID: {jobId}</div>
        <div>Camera ID: {cameraId}</div>
      </div>

      <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
        <div style={{ minWidth: 340 }}>
          <div style={{ marginBottom: 8 }}>
            <label style={{ marginRight: 8 }}>Spot ID:</label>
            <input value={spotId} onChange={(e) => setSpotId(e.target.value)} />
          </div>

          <div style={{ marginBottom: 8 }}>
            <label style={{ marginRight: 8 }}>Type:</label>
            <select value={spotType} onChange={(e) => setSpotType(e.target.value)}>
              <option value="parking">parking</option>
              <option value="loading">loading</option>
              <option value="disabled">disabled</option>
              <option value="ev">ev</option>
              <option value="staff">staff</option>
              <option value="visitor">visitor</option>
            </select>
          </div>

          <div style={{ marginBottom: 8 }}>
            <label style={{ marginRight: 8 }}>Description:</label>
            <input
              style={{ width: 320 }}
              value={spotDesc}
              onChange={(e) => setSpotDesc(e.target.value)}
              placeholder="Near entrance / Loading bay / Staff only"
            />
          </div>

          <div style={{ display: "flex", gap: 8, marginBottom: 10 }}>
            <button onClick={undoPoint}>Undo point</button>
            <button onClick={clearCurrent}>Clear current</button>
            <button onClick={addSpot}>Add Spot</button>
          </div>

          <div style={{ display: "flex", gap: 8, marginBottom: 10 }}>
            <button onClick={removeLastSpot}>Remove last spot</button>
            <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <input
                type="checkbox"
                checked={publish}
                onChange={(e) => setPublish(e.target.checked)}
              />
              Publish immediately
            </label>
          </div>

          <button onClick={saveAndRun} disabled={spots.length === 0}>
            Save Spots & Run Inference
          </button>

          <div style={{ marginTop: 12 }}>{statusMsg}</div>

          {inferenceDone && (
            <div style={{ marginTop: 12 }}>
              <button onClick={goView}>View processed video</button>
            </div>
          )}

          <div style={{ marginTop: 12 }}>
            Current polygon points: {currentPoly.length} <br />
            Total spots saved: {spots.length}
          </div>
        </div>

        <div>
          <canvas
            ref={canvasRef}
            width={960}
            height={540}
            onClick={onCanvasClick}
            style={{ border: "2px solid #444", cursor: "crosshair" }}
          />
        </div>
      </div>
    </div>
  );
}
