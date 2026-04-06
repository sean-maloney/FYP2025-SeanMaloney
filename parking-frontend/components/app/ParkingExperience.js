import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import styles from "../../styles/ParkingExperience.module.css";
import {
  API_BASE,
  getGridConfig,
  getPublishedSpots,
  runParkingExperience,
} from "../../lib/api";

function pointInPolygon(point, polygon, width, height) {
  const x = point[0];
  const y = point[1];
  const pts = polygon.map((p) => [p.x * width, p.y * height]);

  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const xi = pts[i][0];
    const yi = pts[i][1];
    const xj = pts[j][0];
    const yj = pts[j][1];

    const intersect =
      yi > y !== yj > y &&
      x < ((xj - xi) * (y - yi)) / ((yj - yi) || 0.00001) + xi;

    if (intersect) inside = !inside;
  }

  return inside;
}

function centreOfCell(cell, canvasWidth, canvasHeight, rows, cols) {
  const cellWidth = canvasWidth / cols;
  const cellHeight = canvasHeight / rows;

  return [
    cell[1] * cellWidth + cellWidth / 2,
    cell[0] * cellHeight + cellHeight / 2,
  ];
}

export default function ParkingExperience() {
  const canvasRef = useRef(null);
  const imageRef = useRef(null);

  const [cameraId, setCameraId] = useState("cam1");
  const [videoFile, setVideoFile] = useState(null);
  const [adminMode, setAdminMode] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [status, setStatus] = useState("Upload a video and press Get Parking.");
  const [error, setError] = useState("");
  const [snapshotUrl, setSnapshotUrl] = useState("");
  const [spotsConfig, setSpotsConfig] = useState([]);
  const [gridConfig, setGridConfig] = useState(null);
  const [result, setResult] = useState(null);

  const resultStatusMap = useMemo(() => {
    const map = {};
    for (const spot of result?.spots || []) {
      map[spot.id] = spot.status;
    }
    return map;
  }, [result]);

  useEffect(() => {
    if (!snapshotUrl) return;

    const image = new Image();
    image.crossOrigin = "anonymous";

    image.onload = () => {
      imageRef.current = image;
      redraw();
    };

    image.src = snapshotUrl;
  }, [snapshotUrl, spotsConfig, result, gridConfig]);

  useEffect(() => {
    redraw();
  }, [spotsConfig, result, gridConfig, adminMode]);

  async function preloadConfigs(targetCameraId) {
    const [spots, grid] = await Promise.allSettled([
      getPublishedSpots(targetCameraId),
      getGridConfig(targetCameraId),
    ]);

    if (spots.status === "fulfilled") {
      setSpotsConfig(spots.value.spots || []);
    } else {
      setSpotsConfig([]);
    }

    if (grid.status === "fulfilled") {
      setGridConfig(grid.value);
    } else {
      setGridConfig(null);
    }

    return {
      hasPublishedSpots: spots.status === "fulfilled",
      hasGridConfig: grid.status === "fulfilled",
    };
  }

  function redraw() {
    const canvas = canvasRef.current;
    const image = imageRef.current;

    if (!canvas || !image) return;

    const ctx = canvas.getContext("2d");
    canvas.width = image.width;
    canvas.height = image.height;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

    for (const spot of spotsConfig) {
      const points =
        spot.polygon?.map((p) => ({
          x: p.x * canvas.width,
          y: p.y * canvas.height,
        })) || [];

      if (points.length < 3) continue;

      const state = resultStatusMap[spot.id];

      const stroke =
        state === "occupied"
          ? "#ef4444"
          : state === "available"
          ? "#22c55e"
          : "#facc15";

      const fill =
        state === "occupied"
          ? "rgba(239,68,68,0.22)"
          : state === "available"
          ? "rgba(34,197,94,0.22)"
          : "rgba(250,204,21,0.16)";

      ctx.beginPath();
      ctx.moveTo(points[0].x, points[0].y);

      for (let i = 1; i < points.length; i += 1) {
        ctx.lineTo(points[i].x, points[i].y);
      }

      ctx.closePath();
      ctx.fillStyle = fill;
      ctx.strokeStyle = stroke;
      ctx.lineWidth = 3;
      ctx.fill();
      ctx.stroke();

      if (state) {
        ctx.fillStyle = "#ffffff";
        ctx.font = "600 16px Inter, Arial, sans-serif";
        ctx.fillText(
          state === "occupied" ? "Unavailable" : "Available",
          points[0].x + 8,
          points[0].y - 8
        );
      }
    }

    if (gridConfig?.start?.length === 2) {
      const [sx, sy] = centreOfCell(
        gridConfig.start,
        canvas.width,
        canvas.height,
        gridConfig.rows,
        gridConfig.cols
      );

      ctx.beginPath();
      ctx.arc(sx, sy, 10, 0, Math.PI * 2);
      ctx.fillStyle = "#3b82f6";
      ctx.fill();
      ctx.lineWidth = 4;
      ctx.strokeStyle = "#ffffff";
      ctx.stroke();

      ctx.fillStyle = "#ffffff";
      ctx.font = "700 12px Inter, Arial, sans-serif";
      ctx.fillText("Start", sx + 16, sy + 4);
    }

    if ((result?.path || []).length > 1 && gridConfig) {
      ctx.beginPath();
      ctx.lineWidth = 7;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.strokeStyle = "#38bdf8";

      result.path.forEach((cell, index) => {
        const [x, y] = centreOfCell(
          cell,
          canvas.width,
          canvas.height,
          gridConfig.rows,
          gridConfig.cols
        );

        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });

      ctx.stroke();
    }

    if (adminMode && gridConfig?.parking_spaces?.length) {
      ctx.font = "600 12px Inter, Arial, sans-serif";

      for (const cell of gridConfig.parking_spaces) {
        const [x, y] = centreOfCell(
          cell,
          canvas.width,
          canvas.height,
          gridConfig.rows,
          gridConfig.cols
        );

        const linkedSpot = spotsConfig.find((spot) =>
          pointInPolygon([x, y], spot.polygon || [], canvas.width, canvas.height)
        );

        const state = linkedSpot ? resultStatusMap[linkedSpot.id] : null;

        ctx.beginPath();
        ctx.arc(x, y, 7, 0, Math.PI * 2);
        ctx.fillStyle = state === "occupied" ? "#ef4444" : "#22c55e";
        ctx.fill();
      }
    }
  }

  async function handleRunPipeline() {
    if (!videoFile) {
      setError("Please choose a video first.");
      return;
    }

    setIsRunning(true);
    setError("");
    setStatus("Checking admin setup...");

    try {
      await preloadConfigs(cameraId);

      setStatus("Running parking pipeline...");
      const data = await runParkingExperience({ file: videoFile, cameraId });
      console.log("experience result:", data);

      setResult(data);
      setSnapshotUrl(`${API_BASE}${data.snapshot_url}?t=${Date.now()}`);

      setStatus(
        data.route_success
          ? "Nearest available parking spot found."
          : data.route_message || "Parking scan complete."
      );
    } catch (err) {
      setError(err.message || "Something went wrong.");
      setStatus("Pipeline failed.");
    } finally {
      setIsRunning(false);
    }
  }

  const hasJob = !!result?.job_id;

  return (
    <div className={styles.pageShell}>
      <section className={styles.heroCard}>
        <div className={styles.heroTopRow}>
          <div>
            <div className={styles.eyebrow}>Parking detector</div>
            <h1 className={styles.heroTitle}>
              Find the nearest available parking spot
            </h1>
            <p className={styles.heroText}>
              Clean user mode for drivers, with a separate admin toggle for
              calibration and setup.
            </p>
          </div>

          <label className={styles.toggleWrap}>
            <input
              type="checkbox"
              checked={adminMode}
              onChange={(e) => setAdminMode(e.target.checked)}
            />
            <span>Admin mode</span>
          </label>
        </div>

        <div className={styles.centerDial}>
          <button
            className={styles.getParkingButton}
            onClick={handleRunPipeline}
            disabled={isRunning}
          >
            {isRunning ? "Running..." : "Get Parking"}
          </button>
        </div>

        <div className={styles.controlGrid}>
          <label className={styles.fieldBlock}>
            <span>Camera ID</span>
            <input
              value={cameraId}
              onChange={(e) => setCameraId(e.target.value)}
              className={styles.input}
            />
          </label>

          <label className={styles.fieldBlock}>
            <span>Video input</span>
            <input
              type="file"
              accept="video/*"
              onChange={(e) => setVideoFile(e.target.files?.[0] || null)}
              className={styles.fileInput}
            />
          </label>
        </div>

        <div className={styles.statusRow}>
          <div>
            <strong>Status:</strong> {status}
          </div>
          {videoFile ? (
            <div>
              <strong>Selected:</strong> {videoFile.name}
            </div>
          ) : null}
        </div>

        {error ? <div className={styles.errorBox}>{error}</div> : null}
      </section>

      <section className={styles.resultLayout}>
        <div className={styles.canvasCard}>
          <div className={styles.cardHeader}>
            <div>
              <h2>Car park view</h2>
              <p>
                Drivers only see available, unavailable, and the route from the
                start point.
              </p>
            </div>

            {result ? (
              <div className={styles.metricGroup}>
                <div className={styles.metric}>
                  <span>Available</span>
                  <strong>{result.available}</strong>
                </div>
                <div className={styles.metric}>
                  <span>Unavailable</span>
                  <strong>{result.occupied}</strong>
                </div>
              </div>
            ) : null}
          </div>

          <div className={styles.canvasWrap}>
            {snapshotUrl ? (
              <canvas ref={canvasRef} className={styles.canvas} />
            ) : (
              <div className={styles.emptyState}>
                Run the pipeline to show the parking image and route overlay.
              </div>
            )}
          </div>
        </div>

        <aside className={styles.sidePanel}>
          <div className={styles.legendCard}>
            <h3>Legend</h3>
            <div className={styles.legendItem}>
              <span className={`${styles.legendSwatch} ${styles.available}`}></span>
              Available
            </div>
            <div className={styles.legendItem}>
              <span className={`${styles.legendSwatch} ${styles.unavailable}`}></span>
              Unavailable
            </div>
            <div className={styles.legendItem}>
              <span className={`${styles.legendSwatch} ${styles.route}`}></span>
              Nearest route
            </div>
            <div className={styles.legendItem}>
              <span className={`${styles.legendSwatch} ${styles.start}`}></span>
              Start
            </div>
          </div>

          {adminMode ? (
            <div className={styles.adminCard}>
              <h3>Admin tools</h3>
              <p>
                All of your older project pages are available here so admin mode
                becomes the setup hub.
              </p>

              <div className={styles.adminSection}>
                <div className={styles.adminSectionTitle}>Setup tools</div>

                <div className={styles.adminLinks}>
                  <Link href="/upload" className={styles.secondaryLink}>
                    Open Upload Page
                  </Link>

                  <Link href="/camera-monitor" className={styles.secondaryLink}>
                    Open Camera Monitor
                  </Link>

                  <Link href="/pathfinder" className={styles.secondaryLink}>
                    Open Pathfinder Grid Editor
                  </Link>

                  {hasJob ? (
                    <Link
                      href={`/spot-calibration?job_id=${result.job_id}&camera_id=${cameraId}`}
                      className={styles.secondaryLink}
                    >
                      Open Spot Calibration
                    </Link>
                  ) : (
                    <div className={styles.disabledLink}>
                      Run once to unlock Spot Calibration
                    </div>
                  )}
                </div>
              </div>

              <div className={styles.adminSection}>
                <div className={styles.adminSectionTitle}>Job tools</div>

                <div className={styles.adminLinks}>
                  {hasJob ? (
                    <Link
                      href={`/view/${result.job_id}`}
                      className={styles.secondaryLink}
                    >
                      View Processed Video
                    </Link>
                  ) : (
                    <div className={styles.disabledLink}>
                      Run once to unlock Processed Video
                    </div>
                  )}

                  {hasJob ? (
                    <a
                      href={`${API_BASE}/api/jobs/${result.job_id}/snapshot`}
                      target="_blank"
                      rel="noreferrer"
                      className={styles.secondaryLink}
                    >
                      Open Snapshot
                    </a>
                  ) : (
                    <div className={styles.disabledLink}>
                      Run once to unlock Snapshot
                    </div>
                  )}
                </div>
              </div>

              <div className={styles.adminSection}>
                <div className={styles.adminSectionTitle}>Current context</div>

                <div className={styles.adminInfoCard}>
                  <div className={styles.infoRow}>
                    <span>Camera</span>
                    <strong>{cameraId}</strong>
                  </div>
                  <div className={styles.infoRow}>
                    <span>Latest Job</span>
                    <strong>{result?.job_id || "None yet"}</strong>
                  </div>
                  <div className={styles.infoRow}>
                    <span>Snapshot</span>
                    <strong>{snapshotUrl ? "Loaded" : "Not loaded"}</strong>
                  </div>
                </div>
              </div>
            </div>
          ) : null}
        </aside>
      </section>
    </div>
  );
}