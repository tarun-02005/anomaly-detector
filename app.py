from flask import Flask, render_template, request, jsonify, url_for
import os
import cv2
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__, static_folder='static', template_folder='templates')

model = YOLO('best_anomaly_model.pt')

UPLOAD_FOLDER = 'static/uploads/'
PROCESSED_FOLDER = 'static/processed/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detector')
def detector():
    return render_template('detector.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid or no selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    ext = filename.rsplit('.', 1)[1].lower()

    if ext in {'jpg', 'jpeg', 'png'}:
        return process_image(filepath, filename)
    elif ext in {'mp4', 'avi', 'mov'}:
        return process_video(filepath, filename)
    else:
        return jsonify({'error': 'Unsupported file type'}), 400

def process_image(filepath, filename):
    results = model(filepath)
    img = cv2.imread(filepath)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = f"{model.names.get(cls, str(cls))}: {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2)

    processed_path = os.path.join(PROCESSED_FOLDER, filename)
    cv2.imwrite(processed_path, img)
    return jsonify({'type': 'image', 'path': url_for('static', filename=f'processed/{filename}')})

def process_video(filepath, filename):
    cap = cv2.VideoCapture(filepath)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    processed_name = filename.rsplit('.', 1)[0] + '_processed.mp4'
    processed_path = os.path.join(PROCESSED_FOLDER, processed_name)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
    out = cv2.VideoWriter(processed_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = f"{model.names.get(cls, str(cls))}: {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 2)

        out.write(frame)

    cap.release()
    out.release()

    return jsonify({
    'type': 'video',
    'path': url_for('static', filename=f'processed/{processed_name}')  # Example: /static/processed/video_processed.mp4
})


if __name__ == "__main__":
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    port = int(os.getenv('PORT', 10000))
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
