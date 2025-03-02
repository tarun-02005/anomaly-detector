from flask import Flask, render_template, request, jsonify, url_for
import os
import cv2
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__, static_folder='static', template_folder='templates')

# Load YOLOv8 model
model = YOLO('best_anomaly_model.pt')

UPLOAD_FOLDER = 'static/uploads/'
PROCESSED_FOLDER = 'static/processed/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')  # Ensure index.html is in the templates folder

@app.route('/detector')
def detector():
    return render_template('detector.html')  # Ensure detector.html exists in templates

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    results = model(filepath)

    img = cv2.imread(filepath)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{model.names[cls]}: {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    processed_filepath = os.path.join(PROCESSED_FOLDER, filename)
    cv2.imwrite(processed_filepath, img)

    return jsonify({'processed_image': processed_filepath})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)

