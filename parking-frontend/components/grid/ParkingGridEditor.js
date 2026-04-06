import { useEffect, useRef, useState } from "react";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";

const colours = {
  empty: "rgba(0,0,0,0)",
  road: "rgba(34, 197, 94, 0.20)",
  blocked: "rgba(239, 68, 68, 0.35)",
  start: "rgba(59, 130, 246, 0.35)",
  parking: "rgba(234, 179, 8, 0.35)",
};

function createEmptyGrid(targetRows, targetCols) {
  return Array.from({ length: targetRows }, () =>
    Array.from({ length: targetCols }, () => "empty")
  );
}

export default function ParkingGridEditor({
  initialCameraId = "cam1",
  backgroundImageUrl = "",
  initialRows = 12,
  initialCols = 16,
}) {
  const canvasRef = useRef(null);
  const imageRef = useRef(null);

  const [cameraId, setCameraId] = useState(initialCameraId);
  const [rowsInput, setRowsInput] = useState(initialRows);
  const [colsInput, setColsInput] = useState(initialCols);
  const [showGrid, setShowGrid] = useState(true);
  const [selectedTool, setSelectedTool] = useState("road");
  const [goal, setGoal] = useState("");
  const [path, setPath] = useState([]);
  const [selectedGoal, setSelectedGoal] = useState(null);
  const [statusMessage, setStatusMessage] = useState("No route generated yet.");
  const [errorMessage, setErrorMessage] = useState("");
  const [imageNaturalSize, setImageNaturalSize] = useState({
    width: 0,
    height: 0,
  });

  const [grid, setGrid] = useState(() =>
    createEmptyGrid(initialRows, initialCols)
  );

  useEffect(() => {
    setCameraId(initialCameraId || "cam1");
  }, [initialCameraId]);

  const actualRows = grid.length;
  const actualCols = grid[0]?.length || 0;

  useEffect(() => {
    if (!backgroundImageUrl) {
      imageRef.current = null;
      setImageNaturalSize({ width: 0, height: 0 });
      drawCanvas(grid, [], null, 0, 0);
      return;
    }

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = backgroundImageUrl;

    img.onload = () => {
      const width = img.naturalWidth || img.width || 0;
      const height = img.naturalHeight || img.height || 0;
      imageRef.current = img;
      setImageNaturalSize({ width, height });
      drawCanvas(grid, path, selectedGoal, width, height);
    };

    img.onerror = () => {
      imageRef.current = null;
      setErrorMessage("Could not load the selected background image.");
      drawCanvas(grid, path, selectedGoal, 0, 0);
    };
  }, [backgroundImageUrl]);

  useEffect(() => {
    drawCanvas(
      grid,
      path,
      selectedGoal,
      imageNaturalSize.width,
      imageNaturalSize.height
    );
  }, [grid, showGrid, path, selectedGoal, imageNaturalSize]);

  function getCanvasSize(imageWidth, imageHeight, cols, rows) {
    if (imageWidth > 0 && imageHeight > 0) {
      return {
        canvasWidth: imageWidth,
        canvasHeight: imageHeight,
      };
    }

    return {
      canvasWidth: Math.max(cols, 1) * 50,
      canvasHeight: Math.max(rows, 1) * 50,
    };
  }

  function getCellDimensions(imageWidth, imageHeight, cols, rows) {
    const { canvasWidth, canvasHeight } = getCanvasSize(
      imageWidth,
      imageHeight,
      cols,
      rows
    );

    return {
      canvasWidth,
      canvasHeight,
      cellWidth: canvasWidth / Math.max(cols, 1),
      cellHeight: canvasHeight / Math.max(rows, 1),
    };
  }

  function getStartCell(currentGrid) {
    for (let row = 0; row < currentGrid.length; row++) {
      for (let col = 0; col < (currentGrid[row]?.length || 0); col++) {
        if (currentGrid[row][col] === "start") {
          return [row, col];
        }
      }
    }
    return null;
  }

  function drawCanvas(currentGrid, currentPath, currentGoal, imageWidth, imageHeight) {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rows = currentGrid.length;
    const cols = currentGrid[0]?.length || 0;

    if (rows === 0 || cols === 0) return;

    const ctx = canvas.getContext("2d");
    const { canvasWidth, canvasHeight, cellWidth, cellHeight } =
      getCellDimensions(imageWidth, imageHeight, cols, rows);

    canvas.width = canvasWidth;
    canvas.height = canvasHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (imageRef.current && imageWidth > 0 && imageHeight > 0) {
      ctx.drawImage(imageRef.current, 0, 0, canvasWidth, canvasHeight);
    } else {
      ctx.fillStyle = "#1f2937";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }

    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        const x = col * cellWidth;
        const y = row * cellHeight;
        const cellType = currentGrid[row]?.[col] || "empty";

        if (cellType !== "empty") {
          ctx.fillStyle = colours[cellType];
          ctx.fillRect(x, y, cellWidth, cellHeight);
        }

        if (showGrid) {
          ctx.strokeStyle = "rgba(17, 24, 39, 0.75)";
          ctx.lineWidth = 1;
          ctx.strokeRect(x, y, cellWidth, cellHeight);
        }
      }
    }

    const startCell = getStartCell(currentGrid);
    if (startCell) {
      const startX = startCell[1] * cellWidth;
      const startY = startCell[0] * cellHeight;

      ctx.strokeStyle = "#1d4ed8";
      ctx.lineWidth = 3;
      ctx.strokeRect(
        startX + 3,
        startY + 3,
        Math.max(cellWidth - 6, 4),
        Math.max(cellHeight - 6, 4)
      );
    }

    if (currentGoal && currentGoal.length === 2) {
      const goalX = currentGoal[1] * cellWidth;
      const goalY = currentGoal[0] * cellHeight;

      ctx.strokeStyle = "#7c3aed";
      ctx.lineWidth = 3;
      ctx.strokeRect(
        goalX + 3,
        goalY + 3,
        Math.max(cellWidth - 6, 4),
        Math.max(cellHeight - 6, 4)
      );
    }

    if (currentPath.length > 0) {
      ctx.beginPath();
      ctx.strokeStyle = "#2563eb";
      ctx.lineWidth = 4;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";

      currentPath.forEach((cell, index) => {
        const x = cell[1] * cellWidth + cellWidth / 2;
        const y = cell[0] * cellHeight + cellHeight / 2;

        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });

      ctx.stroke();

      currentPath.forEach((cell, index) => {
        const x = cell[1] * cellWidth + cellWidth / 2;
        const y = cell[0] * cellHeight + cellHeight / 2;

        ctx.fillStyle =
          index === 0 ? "#1d4ed8" : index === currentPath.length - 1 ? "#7c3aed" : "#2563eb";

        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
      });
    }
  }

  function handleCanvasClick(event) {
    if (actualRows === 0 || actualCols === 0) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const mouseX = (event.clientX - rect.left) * scaleX;
    const mouseY = (event.clientY - rect.top) * scaleY;

    const { cellWidth, cellHeight } = getCellDimensions(
      imageNaturalSize.width,
      imageNaturalSize.height,
      actualCols,
      actualRows
    );

    const col = Math.floor(mouseX / cellWidth);
    const row = Math.floor(mouseY / cellHeight);

    if (row < 0 || row >= actualRows || col < 0 || col >= actualCols) return;

    const nextGrid = grid.map((r) => [...r]);

    if (selectedTool === "start") {
      for (let r = 0; r < nextGrid.length; r++) {
        for (let c = 0; c < (nextGrid[r]?.length || 0); c++) {
          if (nextGrid[r][c] === "start") {
            nextGrid[r][c] = "road";
          }
        }
      }
    }

    if (!nextGrid[row]) return;
    nextGrid[row][col] = selectedTool;

    setGrid(nextGrid);
    setErrorMessage("");
  }

  function buildPayload() {
    let start = [];
    const parking_spaces = [];
    const numericGrid = [];

    for (let row = 0; row < actualRows; row++) {
      const numericRow = [];

      for (let col = 0; col < actualCols; col++) {
        const value = grid[row][col];

        numericRow.push(value === "blocked" ? 1 : 0);

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
      rows: actualRows,
      cols: actualCols,
      grid: numericGrid,
      start,
      parking_spaces,
    };
  }

  function resetPathState() {
    setPath([]);
    setSelectedGoal(null);
  }

  function applyNewGridSize() {
    const safeRows = Math.max(1, Number(rowsInput) || 1);
    const safeCols = Math.max(1, Number(colsInput) || 1);

    const rebuilt = Array.from({ length: safeRows }, (_, row) =>
      Array.from({ length: safeCols }, (_, col) => grid[row]?.[col] || "empty")
    );

    setGrid(rebuilt);
    resetPathState();
    setErrorMessage("");
    setStatusMessage(`Grid resized to ${safeRows} x ${safeCols}.`);
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

    const loadedRows = data.rows;
    const loadedCols = data.cols;

    const rebuilt = Array.from({ length: loadedRows }, () =>
      Array.from({ length: loadedCols }, () => "road")
    );

    for (let row = 0; row < loadedRows; row++) {
      for (let col = 0; col < loadedCols; col++) {
        rebuilt[row][col] = data.grid[row][col] === 1 ? "blocked" : "road";
      }
    }

    if (data.start && data.start.length === 2) {
      rebuilt[data.start[0]][data.start[1]] = "start";
    }

    for (const [row, col] of data.parking_spaces || []) {
      if (rebuilt[row] && rebuilt[row][col] !== undefined) {
        rebuilt[row][col] = "parking";
      }
    }

    setRowsInput(loadedRows);
    setColsInput(loadedCols);
    setGrid(rebuilt);
    resetPathState();
    setStatusMessage(`Grid loaded (${loadedRows} x ${loadedCols}).`);
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
      setErrorMessage(
        data.detail || data.message || "Could not route to nearest available space."
      );
      resetPathState();
      setStatusMessage("Nearest available route failed.");
      return;
    }

    setPath(data.path || []);
    setSelectedGoal(data.goal || null);
    setStatusMessage(
      data.goal && data.goal.length === 2
        ? `Path overlay drawn to nearest available space at (${data.goal[0]}, ${data.goal[1]}).`
        : "Nearest available route drawn."
    );
  }

  function clearGrid() {
    setGrid(createEmptyGrid(actualRows || 1, actualCols || 1));
    resetPathState();
    setErrorMessage("");
    setStatusMessage("Grid cleared.");
  }

  return (
    <div style={{ padding: "20px" }}>
      <h2>Parking Grid Editor</h2>

      <div style={{ marginBottom: "12px", fontSize: "14px" }}>
        <strong>Camera ID:</strong> {cameraId}
      </div>

      {backgroundImageUrl ? (
        <div
          style={{
            marginBottom: "12px",
            padding: "10px 12px",
            border: "1px solid #16a34a",
            background: "#f0fdf4",
            color: "#166534",
            borderRadius: 8,
            fontSize: "14px",
            maxWidth: "860px",
          }}
        >
          Background image loaded. The image keeps its own size, and the grid is fitted on top of it.
          {imageNaturalSize.width > 0 && imageNaturalSize.height > 0
            ? ` Source image: ${imageNaturalSize.width} x ${imageNaturalSize.height}px`
            : ""}
        </div>
      ) : (
        <div
          style={{
            marginBottom: "12px",
            padding: "10px 12px",
            border: "1px solid #f59e0b",
            background: "#fffbeb",
            color: "#92400e",
            borderRadius: 8,
            fontSize: "14px",
            maxWidth: "860px",
          }}
        >
          Load a camera snapshot or upload a video frame first, then edit and save the grid.
        </div>
      )}

      <div
        style={{
          display: "flex",
          gap: "10px",
          flexWrap: "wrap",
          marginBottom: "12px",
          alignItems: "center",
        }}
      >
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

        <label>
          Rows{" "}
          <input
            type="number"
            min="1"
            value={rowsInput}
            onChange={(e) => setRowsInput(Number(e.target.value))}
            style={{ width: 70 }}
          />
        </label>

        <label>
          Cols{" "}
          <input
            type="number"
            min="1"
            value={colsInput}
            onChange={(e) => setColsInput(Number(e.target.value))}
            style={{ width: 70 }}
          />
        </label>

        <button onClick={applyNewGridSize}>Apply Grid Size</button>
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

      <div
        style={{
          display: "flex",
          gap: "16px",
          alignItems: "flex-start",
          flexWrap: "wrap",
        }}
      >
        <div style={{ overflow: "auto", maxWidth: "100%" }}>
          <canvas
            ref={canvasRef}
            onClick={handleCanvasClick}
            style={{
              border: "1px solid #ccc",
              cursor: "pointer",
              maxWidth: "100%",
              height: "auto",
              background: "#fff",
              display: "block",
            }}
          />
        </div>

        <div style={{ fontSize: "14px", minWidth: "260px" }}>
          <div style={{ marginBottom: "8px" }}>
            <strong>Overlay Legend</strong>
          </div>
          <div style={{ marginBottom: "6px" }}>Blue line = A* path</div>
          <div style={{ marginBottom: "6px" }}>Blue box = start cell</div>
          <div style={{ marginBottom: "6px" }}>Purple box = goal cell</div>
          <div style={{ marginBottom: "6px" }}>Red fill = blocked cell</div>
          <div style={{ marginBottom: "6px" }}>Yellow fill = parking cell</div>
          <div style={{ marginBottom: "6px" }}>Green fill = road cell</div>
          <div style={{ marginBottom: "12px" }} />
          <div><strong>Current Grid</strong></div>
          <div>{actualRows} rows x {actualCols} cols</div>
          <div>
            Image size:{" "}
            {imageNaturalSize.width > 0 && imageNaturalSize.height > 0
              ? `${imageNaturalSize.width} x ${imageNaturalSize.height}px`
              : "No image loaded"}
          </div>
          <div>
            Cell size on image:{" "}
            {imageNaturalSize.width > 0 && imageNaturalSize.height > 0
              ? `${(imageNaturalSize.width / Math.max(actualCols, 1)).toFixed(1)}w x ${(imageNaturalSize.height / Math.max(actualRows, 1)).toFixed(1)}h`
              : "N/A"}
          </div>
          <div>{backgroundImageUrl ? "Grid fitted to image" : "No background image loaded"}</div>
        </div>
      </div>
    </div>
  );
}