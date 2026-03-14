from flask import Flask, render_template, request, send_file
from ultralytics import YOLO
import os
import cv2
import torch

# Limit CPU usage for Render free tier
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"

app = Flask(__name__)

# Writable directory for Render
UPLOAD_FOLDER = "/tmp"
INPUT_PATH = os.path.join(UPLOAD_FOLDER, "input.jpg")
RESULT_PATH = os.path.join(UPLOAD_FOLDER, "result.jpg")

# YOLO model path
MODEL_PATH = "yolov8n.pt"

# Lazy loading model (prevents startup crash)
model = None


@app.route("/", methods=["GET", "POST"])
def index():
    global model

    try:
        if request.method == "POST":

            if "image" not in request.files:
                return "No file uploaded"

            file = request.files["image"]

            if file.filename == "":
                return "No selected file"

            # Save uploaded image
            file.save(INPUT_PATH)
            print("Image saved:", INPUT_PATH)

            # Load model only when needed
            if model is None:
                print("Loading YOLO model...")
                model = YOLO(MODEL_PATH)
                print("Model loaded!")

            # Run detection (low memory size for Render)
            results = model.predict(INPUT_PATH, imgsz=128, device="cpu")

            # Plot detection result
            result_img = results[0].plot()

            if result_img is None:
                return "Detection failed"

            # Safe detection handling
            labels = results[0].names
            boxes = []

            if results[0].boxes is not None:
                boxes = results[0].boxes.cls.tolist()

            # Detection flags
            person_detected = False
            bike_detected = False

            for c in boxes:
                if labels[int(c)] == "person":
                    person_detected = True
                if labels[int(c)] == "motorcycle":
                    bike_detected = True

            # Warning text
            if person_detected and bike_detected:
                cv2.putText(
                    result_img,
                    "Possible No Helmet Rider",
                    (40, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

            # Save result image
            cv2.imwrite(RESULT_PATH, result_img)

            return render_template("index.html", image="result.jpg")

        return render_template("index.html", image=None)

    except Exception as e:
        print("ERROR:", e)
        return f"Error occurred: {e}"


@app.route("/result.jpg")
def result_image():
    if not os.path.exists(RESULT_PATH):
        return "Result image not generated"
    return send_file(RESULT_PATH, mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

# from flask import Flask, render_template, request
# from ultralytics import YOLO
# import os
# import cv2
# import torch

# # Limit CPU usage for Render free tier
# torch.set_num_threads(1)
# os.environ["OMP_NUM_THREADS"] = "1"

# app = Flask(__name__)

# # Use /tmp for writable storage in Render
# UPLOAD_FOLDER = "/tmp"
# INPUT_PATH = os.path.join(UPLOAD_FOLDER, "input.jpg")
# RESULT_PATH = os.path.join(UPLOAD_FOLDER, "result.jpg")

# # Model path
# MODEL_PATH = "yolov8n.pt"

# # Load YOLO model once
# print("Loading YOLO model...")
# model = YOLO(MODEL_PATH)
# print("Model loaded successfully!")

# @app.route("/", methods=["GET", "POST"])
# def index():
#     try:
#         if request.method == "POST":

#             if "image" not in request.files:
#                 return "No file uploaded"

#             file = request.files["image"]

#             if file.filename == "":
#                 return "No selected file"

#             # Save uploaded image
#             file.save(INPUT_PATH)

#             # Run detection
#             results = model.predict(INPUT_PATH, imgsz=128, device="cpu")
#             result_img = results[0].plot()

#             # Check detections
#             boxes = results[0].boxes.cls.tolist()
#             labels = results[0].names

#             person_detected = any(labels[int(c)] == "person" for c in boxes)
#             bike_detected = any(labels[int(c)] == "motorcycle" for c in boxes)

#             # Add warning if both present
#             if person_detected and bike_detected:
#                 cv2.putText(
#                     result_img,
#                     "Possible No Helmet Rider",
#                     (40, 50),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     1,
#                     (0, 0, 255),
#                     3
#                 )

#             # Save result
#             cv2.imwrite(RESULT_PATH, result_img)

#             return render_template("index.html", image="result.jpg")

#         return render_template("index.html", image=None)

#     except Exception as e:
#         print("ERROR:", e)
#         return f"Error occurred: {e}"

# # Route to display result image from /tmp
# @app.route("/result.jpg")
# def result_image():
#     from flask import send_file
#     return send_file(RESULT_PATH, mimetype="image/jpeg")

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)

