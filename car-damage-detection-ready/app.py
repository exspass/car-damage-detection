from __future__ import annotations

import os
import uuid
from pathlib import Path

import cv2
import torch
from flask import Flask, render_template, request, url_for
from ultralytics import YOLO
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
RESULT_FOLDER = BASE_DIR / "static" / "results"
MODEL_PATH = Path(os.getenv("YOLO_MODEL_PATH", BASE_DIR / "models" / "best.pt"))
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONFIDENCE", "0.15"))

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

# Illustrative prototype values only; they are not market quotes or professional estimates.
REPAIR_COSTS = {
    "Front-Windscreen-Damage": 450,
    "Headlight-Damage": 350,
    "Major-Rear-Bumper-Dent": 600,
    "Rear-windscreen-Damage": 450,
    "RunningBoard-Dent": 300,
    "Sidemirror-Damage": 250,
    "Signlight-Damage": 150,
    "Taillight-Damage": 300,
    "bonnet-dent": 400,
    "doorouter-dent": 350,
    "doorouter-scratch": 250,
    "fender-dent": 350,
    "front-bumper-dent": 450,
    "front-bumper-scratch": 300,
    "medium-Bodypanel-Dent": 400,
    "paint-chip": 150,
    "paint-trace": 150,
    "pillar-dent": 500,
    "quaterpanel-dent": 500,
    "rear-bumper-dent": 450,
    "rear-bumper-scratch": 300,
    "roof-dent": 500,
}

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"YOLO weights not found at {MODEL_PATH}. "
        "Place your trained best.pt file in the models/ directory or set YOLO_MODEL_PATH."
    )

model = YOLO(str(MODEL_PATH))


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def render_page(**context):
    defaults = {
        "detections": None,
        "original_image": None,
        "result_image": None,
        "total_cost": 0,
        "error": None,
    }
    defaults.update(context)
    return render_template("index.html", **defaults)


def run_inference(frame):
    preferred_device = os.getenv("YOLO_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    try:
        return model.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
            device=preferred_device,
        )[0]
    except Exception as exc:
        if preferred_device == "cpu":
            raise
        app.logger.warning("Inference on %s failed; retrying on CPU: %s", preferred_device, exc)
        return model.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
            device="cpu",
        )[0]


@app.errorhandler(RequestEntityTooLarge)
def handle_large_upload(_error):
    return render_page(error="The image is too large. Maximum upload size is 10 MB."), 413


@app.route("/", methods=["GET"])
def index():
    return render_page()


@app.route("/analyze", methods=["POST"])
def analyze():
    uploaded_file = request.files.get("image")
    if uploaded_file is None:
        return render_page(error="No image was uploaded."), 400
    if not uploaded_file.filename:
        return render_page(error="Please select an image."), 400
    if not allowed_file(uploaded_file.filename):
        return render_page(error="Unsupported file type. Use JPG, JPEG, PNG or WEBP."), 400

    safe_name = secure_filename(uploaded_file.filename)
    extension = Path(safe_name).suffix.lower()
    upload_name = f"{uuid.uuid4().hex}{extension}"
    upload_path = UPLOAD_FOLDER / upload_name
    result_path = None

    try:
        uploaded_file.save(upload_path)
        frame = cv2.imread(str(upload_path))
        if frame is None:
            upload_path.unlink(missing_ok=True)
            return render_page(error="The uploaded file could not be read as an image."), 400

        prediction = run_inference(frame)
        detections = []
        if prediction.boxes is not None:
            for box in prediction.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                detections.append({
                    "class": model.names[class_id],
                    "confidence": round(confidence * 100, 1),
                })

        total_cost = sum(REPAIR_COSTS.get(item["class"], 0) for item in detections)
        result_name = f"result_{uuid.uuid4().hex}.jpg"
        result_path = RESULT_FOLDER / result_name
        if not cv2.imwrite(str(result_path), prediction.plot()):
            raise RuntimeError("Could not save the annotated image.")

        return render_page(
            detections=detections,
            original_image=url_for("static", filename=f"uploads/{upload_name}"),
            result_image=url_for("static", filename=f"results/{result_name}"),
            total_cost=total_cost,
        )
    except Exception:
        app.logger.exception("Image analysis failed")
        upload_path.unlink(missing_ok=True)
        if result_path is not None:
            result_path.unlink(missing_ok=True)
        return render_page(error="Image analysis failed. Please try another image."), 500


if __name__ == "__main__":
    # Local development only. Use a production WSGI server for deployment.
    app.run(host="127.0.0.1", port=5000, debug=False)
