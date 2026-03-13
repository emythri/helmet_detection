from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import os
import torch
import urllib.request

print("Starting application...")

torch.set_num_threads(1)

app = Flask(__name__)
os.makedirs("static", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Lazy-loaded model
model = None
MODEL_PATH = "models/yolov8n.pt"

def load_model():
    global model

    if model is None:
        # If model file doesn't exist, download manually
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
            if "image" not in request.files:
                return "No file uploaded"

            file = request.files["image"]
            if file.filename == "":
                return "No selected file"

            input_path = os.path.join("static", "input.jpg")
            file.save(input_path)

            yolo_model = load_model()

            # Run detection
            results = yolo_model.predict(input_path, imgsz=320, device="cpu")
            img = results[0].plot()

            # Extract boxes
            boxes = results[0].boxes.cls.tolist()
            labels = results[0].names

            person = False
            bike = False
            for c in boxes:
                name = labels[int(c)]
                if name == "person":
                    person = True
                if name == "motorcycle":
                    bike = True

            if person and bike:
                cv2.putText(
                    img,
                    "Possible No Helmet Rider",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

            result_path = os.path.join("static", "result.jpg")
            cv2.imwrite(result_path, img)

            return render_template("index.html", image="result.jpg")

        return render_template("index.html", image=None)

    except Exception as e:
        print("ERROR:", e)
        return f"Error occurred: {e}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
