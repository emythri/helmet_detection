from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import os
import torch

# Reduce CPU threads (important for Render free plan)
torch.set_num_threads(1)

app = Flask(__name__)

# Ensure static folder exists
os.makedirs("static", exist_ok=True)

model = None  # Load model only when needed


@app.route("/", methods=["GET", "POST"])
def index():

    global model

    if request.method == "POST":

        if "image" not in request.files:
            return "No file uploaded"

        file = request.files["image"]

        if file.filename == "":
            return "No selected file"

        # Save uploaded image
        input_path = os.path.join("static", "input.jpg")
        file.save(input_path)

        # Load YOLO model only when needed
        if model is None:
            model = YOLO("yolov8n.pt")

        # Run detection
        results = model(input_path)

        img = results[0].plot()

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

        # Display warning
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


if __name__ == "__main__":
    app.run(debug=True)
