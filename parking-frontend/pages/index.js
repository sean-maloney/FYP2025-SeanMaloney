import Link from "next/link";

export default function HomePage() {
  const buttonStyle = {
    background: "white",
    color: "black",
    border: "1px solid black",
    borderRadius: 8,
    padding: "6px 12px",
    cursor: "pointer",
    fontSize: 14,
  };

  const primaryButtonStyle = {
    ...buttonStyle,
    background: "black",
    color: "white",
    fontWeight: "bold",
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#e9e9e9",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontFamily: "Arial, sans-serif",
      }}
    >
      <div
        style={{
          background: "white",
          padding: 30,
          borderRadius: 12,
          boxShadow: "0 4px 12px rgba(0,0,0,0.12)",
          textAlign: "center",
          minWidth: 340,
        }}
      >
        <h1
          style={{
            marginTop: 0,
            marginBottom: 20,
            fontSize: 30,
            fontWeight: "bold",
          }}
        >
          Parking Detector
        </h1>

        <div
          style={{
            display: "flex",
            gap: 10,
            justifyContent: "center",
            flexWrap: "wrap",
          }}
        >
          <Link href="/upload">
            <button style={primaryButtonStyle}>Upload Video</button>
          </Link>

          <Link href="/view/jobold">
            <button style={buttonStyle}>View Output (example)</button>
          </Link>

          <Link href="/camera-monitor">
            <button style={buttonStyle}>Live Camera Monitor</button>
          </Link>

          <Link href="/pathfinder">
            <button style={buttonStyle}>Pathfinder</button>
          </Link>
        </div>
      </div>
    </div>
  );
}