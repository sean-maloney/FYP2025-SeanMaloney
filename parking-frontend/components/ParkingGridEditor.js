import { useEffect, useRef, useState } from "react";

const CELL_SIZE = 30;
const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";

const colours = {
  empty: "rgba(0,0,0,0)",
  road: "rgba(34, 197, 94, 0.20)",
  blocked: "rgba(239, 68, 68, 0.35)",
  start: "rgba(59, 130, 246, 0.35)",
  parking: "rgba(234, 179, 8, 0.35)",
};

export default function ParkingGridEditor() {
  const canvasRef = useRef(null);
  const imageRef = useRef(null);

  const [cameraId, setCameraId] = useState("cam1");
  const [rows] = useState(12);
  const [cols] = useState(16);
  const [showGrid, setShowGrid] = useState(true);
  const [selectedTool, setSelectedTool] = useState("road");
  const [goal, setGoal] = useState("");
  const [path, setPath] = useState([]);
  const [selectedGoal, setSelectedGoal] = useState(null);
  const [statusMessage, setStatusMessage] = useState("No route generated yet.");
  const [errorMessage, setErrorMessage] = useState("");

  const [grid, setGrid] = useState(
    Array.from({ length: 12 }, () => Array.from({ length: 16 }, () => "empty"))
  );

  useEffect(() => {
    const img = new Image();
    img.src = "/parking-layout.png";

    img.onload = () => {
      imageRef.current = img;
      drawCanvas(grid, [], null);
    };
  }, []);

  useEffect(() => {
    drawCanvas(grid, path, selectedGoal);
  }, [grid, showGrid, path, selectedGoal]);

  function getStartCell(currentGrid) {
    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        if (currentGrid[row][col] === "start") {
          return [row, col];
        }
      }
    }
    return null;
  }

  function drawCanvas(currentGrid, currentPath, currentGoal) {
    const canvas = canvasRef.current;
    if (!canvas || !imageRef.current) return;

    const ctx = canvas.getContext("2d");
    canvas.width = cols * CELL_SIZE;
    canvas.height = rows * CELL_SIZE;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(imageRef.current, 0, 0, canvas.width, canvas.height);

    if (showGrid) {
      for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
          const x = col * CELL_SIZE;
          const y = row * CELL_SIZE;

          ctx.strokeStyle = "#111827";
          ctx.lineWidth = 1;
          ctx.strokeRect(x, y, CELL_SIZE, CELL_SIZE);

          const cellType = currentGrid[row][col];
          if (cellType !== "empty") {
            ctx.fillStyle = colours[cellType];
            ctx.fillRect(x, y, CELL_SIZE, CELL_SIZE);
          }
        }
      }
    }

    const startCell = getStartCell(currentGrid);
    if (startCell) {
      const startX = startCell[1] * CELL_SIZE;
      const startY = startCell[0] * CELL_SIZE;

      ctx.strokeStyle = "#1d4ed8";
      ctx.lineWidth = 3;
      ctx.strokeRect(startX + 3, startY + 3, CELL_SIZE - 6, CELL_SIZE - 6);
    }

    if (currentGoal && currentGoal.length === 2) {
      const goalX = currentGoal[1] * CELL_SIZE;
      const goalY = currentGoal[0] * CELL_SIZE;

      ctx.strokeStyle = "#7c3aed";
      ctx.lineWidth = 3;
      ctx.strokeRect(goalX + 3, goalY + 3, CELL_SIZE - 6, CELL_SIZE - 6);
    }

    if (currentPath.length > 0) {
      ctx.beginPath();
      ctx.strokeStyle = "#2563eb";
      ctx.lineWidth = 4;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";

      currentPath.forEach((cell, index) => {
        const x = cell[1] * CELL_SIZE + CELL_SIZE / 2;
        const y = cell[0] * CELL_SIZE + CELL_SIZE / 2;

        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });

      ctx.stroke();

      currentPath.forEach((cell, index) => {
        const x = cell[1] * CELL_SIZE + CELL_SIZE / 2;
        const y = cell[0] * CELL_SIZE + CELL_SIZE / 2;

        if (index === 0) {
          ctx.fillStyle = "#1d4ed8";
        } else if (index === currentPath.length - 1) {
          ctx.fillStyle = "#7c3aed";
        } else {
          ctx.fillStyle = "#2563eb";
        }

        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
      });
    }
  }

  function handleCanvasClick(event) {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();

    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;

    const col = Math.floor(mouseX / CELL_SIZE);
    const row = Math.floor(mouseY / CELL_SIZE);

    if (row < 0 || row >= rows || col < 0 || col >= cols) return;

    const nextGrid = grid.map((r) => [...r]);

    if (selectedTool === "start") {
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          if (nextGrid[r][c] === "start") {
            nextGrid[r][c] = "road";
          }
        }
      }
    }

    nextGrid[row][col] = selectedTool;
    setGrid(nextGrid);
    setErrorMessage("");
  }

  function buildPayload() {
    let start = [];
    const parking_spaces = [];
    const numericGrid = [];

    for (let row = 0; row < rows; row++) {
      const numericRow = [];

      for (let col = 0; col < cols; col++) {
        const value = grid[row][col];

        if (value === "blocked") {
          numericRow.push(1);
        } else {
          numericRow.push(0);
        }

        if (value === "start") {
          start = [row, col];
        }

        if (value === "parking") {
          parking_spaces.push([row, col]);
        }
      }

      numericGrid.push(numericRow);
    }

    return {
      camera_id: cameraId,
      rows,
      cols,
      grid: numericGrid,
      start,
      parking_spaces,
    };
  }

  function resetPathState() {
    setPath([]);
    setSelectedGoal(null);
  }

  async function saveGrid() {
    setErrorMessage("");

    const response = await fetch(`${API_BASE}/api/pathfinder/grid/save`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(buildPayload()),
    });

    const data = await response.json();

    if (!response.ok) {
      setErrorMessage(data.detail || "Could not save grid.");
      setStatusMessage("Save failed.");
      return;
    }

    setStatusMessage(data.message || "Grid saved.");
  }

  async function loadGrid() {
    setErrorMessage("");

    const response = await fetch(`${API_BASE}/api/pathfinder/grid/${cameraId}`);
    const data = await response.json();

    if (!response.ok) {
      setErrorMessage(data.detail || "Could not load grid.");
      setStatusMessage("Load failed.");
      return;
    }

    const rebuilt = Array.from({ length: data.rows }, () =>
      Array.from({ length: data.cols }, () => "road")
    );

    for (let row = 0; row < data.rows; row++) {
      for (let col = 0; col < data.cols; col++) {
        rebuilt[row][col] = data.grid[row][col] === 1 ? "blocked" : "road";
      }
    }

    if (data.start && data.start.length === 2) {
      rebuilt[data.start[0]][data.start[1]] = "start";
    }

    for (const [row, col] of data.parking_spaces || []) {
      rebuilt[row][col] = "parking";
    }

    setGrid(rebuilt);
    resetPathState();
    setStatusMessage("Grid loaded.");
  }

  async function runPathfinder() {
    setErrorMessage("");

    const bits = goal.split(",").map((x) => x.trim());
    if (bits.length !== 2) {
      setErrorMessage("Enter goal as row,col for example 5,7.");
      resetPathState();
      return;
    }

    const goalRow = Number(bits[0]);
    const goalCol = Number(bits[1]);

    if (Number.isNaN(goalRow) || Number.isNaN(goalCol)) {
      setErrorMessage("Goal row and column must both be numbers.");
      resetPathState();
      return;
    }

    const start = buildPayload().start;
    if (!start || start.length !== 2) {
      setErrorMessage("Please set a start cell first.");
      resetPathState();
      return;
    }

    const response = await fetch(`${API_BASE}/api/pathfinder/run/${cameraId}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        start,
        goal: [goalRow, goalCol],
      }),
    });

    const data = await response.json();

    if (!response.ok || !data.success) {
      setErrorMessage(data.detail || data.message || "No path found.");
      resetPathState();
      setStatusMessage("Path generation failed.");
      return;
    }

    setPath(data.path);
    setSelectedGoal([goalRow, goalCol]);
    setStatusMessage(`Path overlay drawn to goal (${goalRow}, ${goalCol}).`);
  }

  async function routeNearestAvailable() {
    setErrorMessage("");

    const start = buildPayload().start;
    if (!start || start.length !== 2) {
      setErrorMessage("Please set a start cell first.");
      resetPathState();
      return;
    }

    const response = await fetch(`${API_BASE}/api/pathfinder/run-nearest/${cameraId}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ start }),
    });

    const data = await response.json();

    if (!response.ok || !data.success) {
      setErrorMessage(data.detail || data.message || "Could not route to nearest available space.");
      resetPathState();
      setStatusMessage("Nearest available route failed.");
      return;
    }

    setPath(data.path || []);
    setSelectedGoal(data.goal || null);

    if (data.goal && data.goal.length === 2) {
      setStatusMessage(`Path overlay drawn to nearest available space at (${data.goal[0]}, ${data.goal[1]}).`);
    } else {
      setStatusMessage("Nearest available route drawn.");
    }
  }

  function clearGrid() {
    setGrid(Array.from({ length: rows }, () => Array.from({ length: cols }, () => "empty")));
    resetPathState();
    setErrorMessage("");
    setStatusMessage("Grid cleared.");
  }

  return (
    <div style={{ padding: "20px" }}>
      <h2>Parking Grid Editor</h2>

      <div style={{ display: "flex", gap: "10px", flexWrap: "wrap", marginBottom: "12px" }}>
        <input
          value={cameraId}
          onChange={(e) => setCameraId(e.target.value)}
          placeholder="camera id"
        />

        <select value={selectedTool} onChange={(e) => setSelectedTool(e.target.value)}>
          <option value="empty">Empty</option>
          <option value="road">Road</option>
          <option value="blocked">Blocked</option>
          <option value="start">Start</option>
          <option value="parking">Parking</option>
        </select>

        <button onClick={() => setShowGrid(!showGrid)}>
          {showGrid ? "Hide Grid" : "Show Grid"}
        </button>

        <button onClick={saveGrid}>Save Grid</button>
        <button onClick={loadGrid}>Load Grid</button>
        <button onClick={clearGrid}>Clear Grid</button>

        <input
          value={goal}
          onChange={(e) => setGoal(e.target.value)}
          placeholder="goal row,col"
        />

        <button onClick={runPathfinder}>Run A*</button>
        <button onClick={routeNearestAvailable}>Route Nearest Available</button>
      </div>

      <div style={{ marginBottom: "8px", fontSize: "14px" }}>
        <strong>Status:</strong> {statusMessage}
      </div>

      {errorMessage ? (
        <div
          style={{
            marginBottom: "12px",
            padding: "10px 12px",
            border: "1px solid #ef4444",
            background: "#fef2f2",
            color: "#b91c1c",
            borderRadius: 8,
            fontSize: "14px",
            maxWidth: "700px",
          }}
        >
          <strong>Error:</strong> {errorMessage}
        </div>
      ) : null}

      <div style={{ display: "flex", gap: "16px", alignItems: "flex-start", flexWrap: "wrap" }}>
        <canvas
          ref={canvasRef}
          onClick={handleCanvasClick}
          style={{
            border: "1px solid #ccc",
            cursor: "pointer",
            maxWidth: "100%",
            background: "#fff",
          }}
        />

        <div style={{ fontSize: "14px", minWidth: "220px" }}>
          <div style={{ marginBottom: "8px" }}><strong>Overlay Legend</strong></div>
          <div style={{ marginBottom: "6px" }}>Blue line = A* path</div>
          <div style={{ marginBottom: "6px" }}>Blue box = start cell</div>
          <div style={{ marginBottom: "6px" }}>Purple box = goal cell</div>
          <div style={{ marginBottom: "6px" }}>Red fill = blocked cell</div>
          <div style={{ marginBottom: "6px" }}>Yellow fill = parking cell</div>
          <div style={{ marginBottom: "6px" }}>Green fill = road cell</div>
        </div>
      </div>
    </div>
  );
}