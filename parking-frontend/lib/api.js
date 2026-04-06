export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";

async function parseResponse(response) {
  const text = await response.text().catch(() => "");

  let data = {};
  try {
    data = JSON.parse(text);
  } catch {
    if (!response.ok) throw new Error(`HTTP ${response.status}: ${text || "Request failed."}`);
    throw new Error(`Invalid JSON response: ${text.slice(0, 200)}`);
  }

  if (!response.ok) {
    throw new Error(data.detail || data.message || `HTTP ${response.status}`);
  }

  return data;
}

export async function runParkingExperience({ file, cameraId }) {
  const body = new FormData();
  body.append("file", file);
  body.append("camera_id", cameraId);

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 120_000);

  try {
    const response = await fetch(`${API_BASE}/api/experience/run`, {
      method: "POST",
      body,
      signal: controller.signal,
    });
    return parseResponse(response);
  } catch (err) {
    if (err.name === "AbortError") throw new Error("Request timed out after 2 minutes.");
    throw err;
  } finally {
    clearTimeout(timeout);
  }
}

export async function getPublishedSpots(cameraId) {
  const response = await fetch(
    `${API_BASE}/api/cameras/${cameraId}/spots/published`
  );
  return parseResponse(response);
}

export async function getGridConfig(cameraId) {
  const response = await fetch(`${API_BASE}/api/pathfinder/grid/${cameraId}`);
  return parseResponse(response);
}