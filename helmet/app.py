from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import os

app = Flask(__name__)

# Ensure static folder exists
os.makedirs("static", exist_ok=True)

model = YOLO("yolov8n.pt")

@app.route("/", methods=["GET","POST"])
def index():

    if request.method == "POST":

        if "image" not in request.files:
            return "No file uploaded"

        file = request.files["image"]

        if file.filename == "":
            return "No selected file"

        filepath = os.path.join("static", "input.jpg")
        file.save(filepath)

        results = model(filepath)

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

        if person and bike:
            cv2.putText(
                img,
                "Possible No Helmet Rider",
                (50,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                3
            )

        result_path = os.path.join("static", "result.jpg")
        cv2.imwrite(result_path, img)

        return render_template("index.html", image="result.jpg")

    return render_template("index.html", image=None)

if __name__ == "__main__":
    app.run(debug=True)
