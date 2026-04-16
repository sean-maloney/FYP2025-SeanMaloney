import csv
from pathlib import Path
import matplotlib.pyplot as plt

RESULTS_CSV = Path(__file__).parent / "runs/detect/visdrone_parking_detector4/results.csv"
OUTPUT_PNG  = Path(__file__).parent / "runs/detect/visdrone_parking_detector4/performance_summary.png"


def load_results(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k.strip(): float(v) for k, v in row.items()})
    return rows


def main():
    rows = load_results(RESULTS_CSV)

    epochs    = [r["epoch"] for r in rows]
    map50     = [r["metrics/mAP50(B)"] for r in rows]
    precision = [r["metrics/precision(B)"] for r in rows]
    recall    = [r["metrics/recall(B)"] for r in rows]
    best      = max(rows, key=lambda r: r["metrics/mAP50(B)"])

    fig, ax = plt.subplots(figsize=(2.188, 2.234), dpi=300, facecolor="white")
    ax.set_facecolor("white")

    ax.plot(epochs, precision, color="#16a34a", linewidth=1.2, label=f"Precision: {best['metrics/precision(B)']:.3f}\nOf every vehicle the model detected,\nhow many were actually real vehicles?\n")
    ax.plot(epochs, recall,    color="#ea580c", linewidth=1.2, label=f"Recall: {best['metrics/recall(B)']:.3f}\nOf every real vehicle in the image,\nhow many did the model successfully find?\n")
    ax.plot(epochs, map50,     color="#2563eb", linewidth=1.5, label=f"mAP@50: {best['metrics/mAP50(B)']:.3f}\nOverall score — are the detection boxes\nlanding accurately on vehicles?")

    ax.text(0.98, 0.08, "Higher = better", transform=ax.transAxes,
            fontsize=3, color="#9ca3af", ha="right")

    ax.set_xlabel("Training Epoch", color="#374151", fontsize=5)
    ax.set_ylabel("Detection Accuracy (0 = worst, 1 = best)", color="#374151", fontsize=4)
    ax.tick_params(colors="#374151", labelsize=4)
    ax.set_xticks(epochs)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_ticklabels(["0", "0.25", "0.50", "0.75", "1.0"])
    for spine in ax.spines.values():
        spine.set_edgecolor("#d1d5db")
    ax.grid(color="#e5e7eb", linewidth=0.5)
    ax.legend(fontsize=3.5, facecolor="white", labelcolor="#111827", edgecolor="#d1d5db", loc="upper left", bbox_to_anchor=(0, 1), borderaxespad=0.3, handlelength=1.5, handleheight=0.8)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches=None, facecolor="white")
    fig.canvas.draw()
    print(f"Saved to {OUTPUT_PNG} — size: {fig.get_size_inches() * 100} px")


if __name__ == "__main__":
    main()
