from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import cv2
import torch
import urllib.request

# Limit CPU threads for Render
torch.set_num_threads(1)

# Initialize Flask app
app = Flask(__name__)

# Auto-create folders
STATIC_DIR = "static"
MODELS_DIR = "models"
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# YOLO model path
MODEL_PATH = os.path.join(MODELS_DIR, "yolov8n.pt")
model = None  # Lazy load

def load_model():
    global model
    if model is None:
        # Download YOLOv8n weights if missing
        if not os.path.isfile(MODEL_PATH):
            print("Downloading YOLOv8n weights...")
            url = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt"
            urllib.request.urlretrieve(url, MODEL_PATH)
            print("Download complete!")
        print("Loading YOLO model...")
        model = YOLO(MODEL_PATH)
        print("YOLO model loaded successfully")
    return model

@app.route("/", methods=["GET", "POST"])
def index():
    try:
        if request.method == "POST":
            # Check file upload
            if "image" not in request.files:
                return "No file uploaded"
            file = request.files["image"]
            if file.filename == "":
                return "No selected file"

            # Save uploaded file
            input_path = os.path.join(STATIC_DIR, "input.jpg")
            file.save(input_path)

            # Load YOLO model
            yolo_model = load_model()

            # Run detection
            results = yolo_model.predict(input_path, imgsz=320, device="cpu")
            result_img = results[0].plot()

            # Check for person and motorcycle
            boxes = results[0].boxes.cls.tolist()
            labels = results[0].names
            person = any(labels[int(c)] == "person" for c in boxes)
            bike = any(labels[int(c)] == "motorcycle" for c in boxes)

            # Add warning text if both detected
            if person and bike:
                cv2.putText(
                    result_img,
                    "Possible No Helmet Rider",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

            # Save result
            result_path = os.path.join(STATIC_DIR, "result.jpg")
            cv2.imwrite(result_path, result_img)

            return render_template("index.html", image="result.jpg")

        # GET request
        return render_template("index.html", image=None)

    except Exception as e:
        print("ERROR:", e)
        return f"Error occurred: {e}"

if __name__ == "__main__":
    # Gunicorn will handle the port on Render
    app.run(host="0.0.0.0")
