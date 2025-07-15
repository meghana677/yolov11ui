from flask import Flask, render_template, request, redirect, url_for
import os
from ultralytics import YOLO
import cv2
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load your YOLOv11 model (custom or pre-trained)
model = YOLO("yolo11m.pt")  # Change this path if you have a custom .pt file

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload/image', methods=['POST'])
def upload_image():
    file = request.files['image']
    if file:
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4().hex}_input.jpg")
        file.save(input_path)

        # Inference
        results = model(input_path)
        result_img = results[0].plot()

        output_path = input_path.replace("_input", "_output")
        cv2.imwrite(output_path, result_img)

        return render_template('result_image.html', input_image=input_path, output_image=output_path)
    return redirect(url_for('index'))

@app.route('/upload/video', methods=['POST'])
def upload_video():
    file = request.files['video']
    if file:
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4().hex}_input.mp4")
        file.save(input_path)

        output_path = input_path.replace("_input", "_output")
        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            result_frame = results[0].plot()
            out.write(result_frame)

        cap.release()
        out.release()

        return render_template('result_video.html', input_video=input_path, output_video=output_path)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
