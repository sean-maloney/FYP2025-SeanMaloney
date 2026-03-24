import { useEffect, useRef, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";

function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}

function toNormPoint(e, canvas) {
  const rect = canvas.getBoundingClientRect();
  const xPx = e.clientX - rect.left;
  const yPx = e.clientY - rect.top;
  const x = clamp01(xPx / rect.width);
  const y = clamp01(yPx / rect.height);
  return { x, y };
}

export default function CameraMonitorPage() {
  const canvasRef = useRef(null);
  const intervalRef = useRef(null);

  const [cameraId, setCameraId] = useState("cam1");
  const [imageUrl, setImageUrl] = useState("");
  const [imageObj, setImageObj] = useState(null);

  const [currentPoly, setCurrentPoly] = useState([]);
  const [spots, setSpots] = useState([]);
  const [spotId, setSpotId] = useState("A01");
  const [spotType, setSpotType] = useState("parking");
  const [spotDesc, setSpotDesc] = useState("");

  const [statusMsg, setStatusMsg] = useState("");
  const [available, setAvailable] = useState(null);
  const [occupied, setOccupied] = useState(null);
  const [liveSpots, setLiveSpots] = useState([]);
  const [monitoring, setMonitoring] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(false);

  useEffect(() => {
    if (!imageUrl) {
      setImageObj(null);
      return;
    }
    const img = new Image();
    img.onload = () => setImageObj(img);
    img.onerror = () => setImageObj(null);
    img.src = imageUrl;
  }, [imageUrl]);

  useEffect(() => {
    redraw();
  }, [imageObj, spots, currentPoly, liveSpots]);

  useEffect(() => {
    loadPublishedSpots();
    return () => {
      stopMonitoring();
    };
  }, [cameraId]);

  useEffect(() => {
    if (autoRefresh && spots.length > 0) {
      startMonitoring();
    } else {
      stopMonitoring();
    }
  }, [autoRefresh]);

  function loadImageUrl() {
    return `${API_BASE}/api/cameras/${cameraId}/image?t=${Date.now()}`;
  }

  async function loadPublishedSpots() {
    try {
      const res = await fetch(`${API_BASE}/api/cameras/${cameraId}/spots/published`);
      if (!res.ok) return;
      const data = await res.json();
      setSpots(data.spots || []);
    } catch (_) {}
  }

  function redraw() {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);

    if (imageObj) {
      ctx.drawImage(imageObj, 0, 0, w, h);
    } else {
      ctx.fillStyle = "#222";
      ctx.fillRect(0, 0, w, h);
    }

    const statusMap = {};
    for (const s of liveSpots) statusMap[s.id] = s.status;

    spots.forEach((s) => {
      const pts = s.polygon.map((p) => ({ x: p.x * w, y: p.y * h }));
      if (pts.length < 3) return;

      let stroke = "yellow";
      let fill = "rgba(255,255,0,0.10)";

      if (statusMap[s.id] === "occupied") {
        stroke = "red";
        fill = "rgba(255,0,0,0.12)";
      } else if (statusMap[s.id] === "available") {
        stroke = "lime";
        fill = "rgba(0,255,0,0.12)";
      }

      ctx.lineWidth = 2;
      ctx.strokeStyle = stroke;
      ctx.fillStyle = fill;
      ctx.font = "16px Arial";

      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = stroke;
      ctx.fillText(`${s.id} (${s.type})`, pts[0].x + 6, pts[0].y - 6);
    });

    if (currentPoly.length > 0) {
      const pts = currentPoly.map((p) => ({ x: p.x * w, y: p.y * h }));

      ctx.strokeStyle = "cyan";
      ctx.fillStyle = "rgba(0,255,255,0.10)";
      ctx.lineWidth = 2;

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
  }

  async function getCameraPhoto() {
    setStatusMsg("Getting camera photo...");
    try {
      const res = await fetch(`${API_BASE}/api/cameras/${cameraId}/capture`, {
        method: "POST",
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to get camera photo");
      setImageUrl(loadImageUrl());
      setStatusMsg("Camera photo loaded");
    } catch (e) {
      setStatusMsg(e.message);
    }
  }

  function onCanvasClick(e) {
    const canvas = canvasRef.current;
    if (!canvas || !imageObj) return;
    const p = toNormPoint(e, canvas);
    setCurrentPoly((prev) => [...prev, p]);
  }

  function undoPoint() {
    setCurrentPoly((prev) => prev.slice(0, -1));
  }

  function clearCurrent() {
    setCurrentPoly([]);
  }

  function addSpot() {
    if (currentPoly.length < 3) {
      setStatusMsg("Need at least 3 points to make a spot");
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
  }

  function removeLastSpot() {
    setSpots((prev) => prev.slice(0, -1));
  }

  async function saveSpots() {
    setStatusMsg("Saving spots...");
    try {
      const payload = {
        camera_id: cameraId,
        frame_size: { w: 1280, h: 720 },
        spots,
      };

      const res = await fetch(`${API_BASE}/api/cameras/${cameraId}/spots?publish=true`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || `HTTP ${res.status}`);
      }

      const data = await res.json();
      setStatusMsg(`Saved version ${data.version}`);
    } catch (e) {
      setStatusMsg(`Save failed ${e.message}`);
    }
  }

  async function refreshNow() {
    try {
      const res = await fetch(`${API_BASE}/api/cameras/${cameraId}/refresh`, {
        method: "POST",
      });

      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || `HTTP ${res.status}`);
      }

      const data = await res.json();
      setImageUrl(loadImageUrl());
      setAvailable(data.available);
      setOccupied(data.occupied);
      setLiveSpots(data.spots || []);
      setStatusMsg(`Updated ${new Date().toLocaleTimeString()}`);
    } catch (e) {
      setStatusMsg(`Refresh failed ${e.message}`);
    }
  }

  function startMonitoring() {
    if (spots.length === 0) {
      setStatusMsg("Save spots first");
      return;
    }
    stopMonitoring();
    setMonitoring(true);
    refreshNow();
    intervalRef.current = setInterval(refreshNow, 30000);
  }

  function stopMonitoring() {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setMonitoring(false);
  }

  function onToggleAutoRefresh(e) {
    const checked = e.target.checked;
    setAutoRefresh(checked);
    if (checked && spots.length === 0) {
      setStatusMsg("Save spots first before enabling auto refresh");
      setAutoRefresh(false);
    }
  }

  return (
    <div style={{ padding: 16 }}>
      <h2>Camera Monitor</h2>

      <div style={{ display: "flex", gap: 8, marginBottom: 12, flexWrap: "wrap", alignItems: "center" }}>
        <label>Camera ID:</label>
        <input value={cameraId} onChange={(e) => setCameraId(e.target.value)} />
        <button onClick={getCameraPhoto}>Get Camera Photo</button>
        <button onClick={saveSpots} disabled={spots.length === 0}>Save Spots</button>
        <button onClick={refreshNow} disabled={spots.length === 0}>Refresh Now</button>
        <label style={{ display: "flex", alignItems: "center", gap: 6, marginLeft: 8 }}>
          <input type="checkbox" checked={autoRefresh} onChange={onToggleAutoRefresh} />
          Auto Refresh Every 30s
        </label>
      </div>

      <div style={{ marginBottom: 10 }}>{statusMsg}</div>

      <div style={{ marginBottom: 10 }}>
        <strong>Auto Refresh:</strong> {autoRefresh ? "On" : "Off"}{" "}
        <strong style={{ marginLeft: 16 }}>Monitoring:</strong> {monitoring ? "On" : "Off"}{" "}
        <strong style={{ marginLeft: 16 }}>Available:</strong> {available ?? "-"}{" "}
        <strong style={{ marginLeft: 16 }}>Occupied:</strong> {occupied ?? "-"}
      </div>

      <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
        <div style={{ minWidth: 320 }}>
          <div style={{ marginBottom: 8 }}>
            <label>Spot ID: </label>
            <input value={spotId} onChange={(e) => setSpotId(e.target.value)} />
          </div>

          <div style={{ marginBottom: 8 }}>
            <label>Type: </label>
            <select value={spotType} onChange={(e) => setSpotType(e.target.value)}>
              <option value="parking">parking</option>
              <option value="handicap">handicap</option>
              <option value="loading">loading</option>
              <option value="staff">staff</option>
              <option value="other">other</option>
            </select>
          </div>

          <div style={{ marginBottom: 8 }}>
            <label>Description: </label>
            <input
              value={spotDesc}
              onChange={(e) => setSpotDesc(e.target.value)}
              placeholder="Near entrance / staff / loading bay"
              style={{ width: 260 }}
            />
          </div>

          <div style={{ display: "flex", gap: 8, marginBottom: 8, flexWrap: "wrap" }}>
            <button onClick={undoPoint}>Undo point</button>
            <button onClick={clearCurrent}>Clear current</button>
            <button onClick={addSpot}>Add Spot</button>
            <button onClick={removeLastSpot}>Remove last spot</button>
          </div>

          <div style={{ marginBottom: 12 }}>
            Current points: {currentPoly.length} | Total spots: {spots.length}
          </div>

          <div>
            <h4>Spot Status</h4>
            {spots.map((s) => {
              const live = liveSpots.find((x) => x.id === s.id);
              return (
                <div key={s.id} style={{ marginBottom: 6 }}>
                  <strong>{s.id}</strong> ({s.type}) — {live?.status || "not updated yet"}
                </div>
              );
            })}
          </div>
        </div>

        <div>
          <canvas
            ref={canvasRef}
            width={960}
            height={540}
            onClick={onCanvasClick}
            style={{ border: "2px solid #555", cursor: "crosshair", maxWidth: "100%" }}
          />
        </div>
      </div>
    </div>
  );
}