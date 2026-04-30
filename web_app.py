"""Flask frontend for object detection and segmentation."""

from __future__ import annotations

import uuid
from pathlib import Path

from flask import Flask, render_template, request, url_for
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "web_uploads"
GENERATED_DIR = BASE_DIR / "static" / "generated"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
model = YOLO("yolov8n-seg.pt")


def detect_and_segment(image_path: Path) -> tuple[str, list[tuple[str, float]]]:
    """Run object detection + segmentation and save annotated image."""
    results = model.predict(source=str(image_path), conf=0.25, verbose=False)
    if not results:
        raise ValueError("No result returned by the segmentation model.")

    result = results[0]
    plotted = result.plot()  # includes masks, bounding boxes, and labels

    detections: list[tuple[str, float]] = []
    if result.boxes is not None and len(result.boxes) > 0:
        names = result.names
        for box in result.boxes:
            cls_idx = int(box.cls[0])
            conf = float(box.conf[0]) * 100
            detections.append((str(names[cls_idx]), conf))
        detections.sort(key=lambda x: x[1], reverse=True)

    output_name = f"annotated_{uuid.uuid4().hex}.jpg"
    output_path = GENERATED_DIR / output_name
    import cv2

    cv2.imwrite(str(output_path), plotted)
    return output_name, detections


@app.route("/", methods=["GET", "POST"])
def index():
    context = {"detections": None, "annotated_url": None, "error": None}

    if request.method == "POST":
        file = request.files.get("image")
        if file is None or file.filename == "":
            context["error"] = "Please choose an image file."
            return render_template("index.html", **context)

        suffix = Path(file.filename).suffix.lower() or ".jpg"
        filename = f"upload_{uuid.uuid4().hex}{suffix}"
        upload_path = UPLOAD_DIR / filename
        file.save(upload_path)

        try:
            output_name, detections = detect_and_segment(upload_path)
            context["detections"] = detections
            context["annotated_url"] = url_for("static", filename=f"generated/{output_name}")
        except Exception as exc:  # pylint: disable=broad-except
            context["error"] = f"Could not process image: {exc}"

    return render_template("index.html", **context)


if __name__ == "__main__":
    app.run(debug=True)
