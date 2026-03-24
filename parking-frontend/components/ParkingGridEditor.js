import { useEffect, useRef, useState } from "react";

const CELL_SIZE = 30;
const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";

const colours = {
  empty: "rgba(0,0,0,0)",
  road: "rgba(34, 197, 94, 0.25)",
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

  const [grid, setGrid] = useState(
    Array.from({ length: 12 }, () => Array.from({ length: 16 }, () => "empty"))
  );

  useEffect(() => {
    const img = new Image();
    img.src = "/parking-layout.png";
    img.onload = () => {
      imageRef.current = img;
      drawCanvas(grid, []);
    };
  }, []);

  useEffect(() => {
    drawCanvas(grid, path);
  }, [grid, showGrid, path]);

  function drawCanvas(currentGrid, currentPath) {
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
          ctx.strokeRect(x, y, CELL_SIZE, CELL_SIZE);

          const cellType = currentGrid[row][col];
          if (cellType !== "empty") {
            ctx.fillStyle = colours[cellType];
            ctx.fillRect(x, y, CELL_SIZE, CELL_SIZE);
          }
        }
      }
    }

    if (currentPath.length > 0) {
      ctx.beginPath();
      ctx.strokeStyle = "#2563eb";
      ctx.lineWidth = 4;

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

  async function saveGrid() {
    const response = await fetch(`${API_BASE}/api/pathfinder/grid/save`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(buildPayload()),
    });

    const data = await response.json();
    alert(data.message || "saved");
  }

  async function loadGrid() {
    const response = await fetch(`${API_BASE}/api/pathfinder/grid/${cameraId}`);
    if (!response.ok) {
      alert("could not load grid");
      return;
    }

    const data = await response.json();
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
    setPath([]);
  }

  async function runPathfinder() {
    const bits = goal.split(",").map((x) => x.trim());
    if (bits.length !== 2) {
      alert("enter goal as row,col for example 5,7");
      return;
    }

    const goalRow = Number(bits[0]);
    const goalCol = Number(bits[1]);

    const start = buildPayload().start;
    if (!start || start.length !== 2) {
      alert("please set a start cell first");
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

    if (!data.success) {
      alert(data.message || "no path found");
      setPath([]);
      return;
    }

    setPath(data.path);
  }

  function clearGrid() {
    setGrid(Array.from({ length: rows }, () => Array.from({ length: cols }, () => "empty")));
    setPath([]);
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
      </div>

      <canvas
        ref={canvasRef}
        onClick={handleCanvasClick}
        style={{ border: "1px solid #ccc", cursor: "pointer", maxWidth: "100%" }}
      />
    </div>
  );
}