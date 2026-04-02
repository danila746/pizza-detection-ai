from flask import Flask, render_template, request, redirect, Response, send_file
import os
import json
from datetime import datetime
from ultralytics import YOLO
from openpyxl import Workbook
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = YOLO("yolov8n.pt")

@app.route("/")
def index():
    if not os.path.exists("history.json"):
        with open("history.json", "w") as f:
            json.dump([], f)

    with open("history.json", "r") as f:
        history = json.load(f)

    return render_template("index.html", history=history)

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]

    if file:
        path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(path)

        results = model(path)[0]

        pizza_count = 0
        for box in results.boxes:
            class_id = int(box.cls[0])
            if model.names[class_id] == "pizza":
                pizza_count += 1

        result_path = os.path.join(app.config["UPLOAD_FOLDER"], "result.jpg")
        results.save(filename=result_path)

        record = {
            "filename": file.filename,
            "count": pizza_count,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        with open("history.json", "r") as f:
            data = json.load(f)

        data.append(record)

        with open("history.json", "w") as f:
            json.dump(data, f, indent=4)

        return redirect("/")

    return redirect("/")

def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)[0]

        pizza_count = 0
        for box in results.boxes:
            class_id = int(box.cls[0])
            if model.names[class_id] == "pizza":
                pizza_count += 1

        annotated = results.plot()

        cv2.putText(annotated, f"Pizzas: {pizza_count}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', annotated)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/download")
def download():
    with open("history.json", "r") as f:
        data = json.load(f)

    wb = Workbook()
    ws = wb.active
    ws.append(["Дата", "Файл", "Количество пицц"])

    for item in data:
        ws.append([item["date"], item["filename"], item["count"]])

    file_path = "report.xlsx"
    wb.save(file_path)

    return send_file(file_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
