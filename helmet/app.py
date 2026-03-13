from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import os
import torch

print("Starting application...")

torch.set_num_threads(1)
print("Torch threads limited")

app = Flask(__name__)
print("Flask app created")

# Ensure static folder exists
os.makedirs("static", exist_ok=True)
print("Static folder checked")

print("Loading YOLO model...")
model = YOLO("yolov8n.pt")
print("YOLO model loaded successfully")


@app.route("/", methods=["GET", "POST"])
def index():

    print("Route '/' accessed")

    try:

        if request.method == "POST":

            print("POST request received")

            if "image" not in request.files:
                print("No image in request")
                return "No file uploaded"

            file = request.files["image"]
            print("Image received")

            if file.filename == "":
                print("Empty filename")
                return "No selected file"

            input_path = os.path.join("static", "input.jpg")
            file.save(input_path)
            print("Image saved to static/input.jpg")

            print("Running YOLO detection...")
            results = model.predict(input_path, imgsz=320, device="cpu")
            print("Detection completed")

            img = results[0].plot()
            print("Result image plotted")

            boxes = results[0].boxes.cls.tolist()
            labels = results[0].names
            print("Boxes extracted")

            person = False
            bike = False

            for c in boxes:
                name = labels[int(c)]

                if name == "person":
                    person = True

                if name == "motorcycle":
                    bike = True

            print("Person detected:", person)
            print("Bike detected:", bike)

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
                print("Warning text added")

            result_path = os.path.join("static", "result.jpg")
            cv2.imwrite(result_path, img)
            print("Result image saved")

            return render_template("index.html", image="result.jpg")

        print("GET request - loading page")
        return render_template("index.html", image=None)

    except Exception as e:
        print("ERROR OCCURRED:", str(e))
        return f"Error occurred: {str(e)}"


if __name__ == "__main__":
    print("Running Flask app...")
    app.run(debug=True)
