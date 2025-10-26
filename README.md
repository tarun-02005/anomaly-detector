# Anomaly Detector

A Flask-based web application for detecting anomalies in images and videos using YOLOv8.

## Features

- 🖼️ Image anomaly detection
- 🎥 Video anomaly detection
- 🚀 Real-time processing with YOLO model
- 📊 Visual bounding boxes with confidence scores

## Tech Stack

- **Backend**: Flask 3.0.3
- **ML Model**: YOLOv8 (Ultralytics)
- **Computer Vision**: OpenCV
- **Deep Learning**: PyTorch

## Local Development

### Prerequisites

- Python 3.12.10
- Virtual environment

### Setup

1. Clone the repository:
```bash
git clone https://github.com/tarun-02005/anomaly-detector.git
cd anomaly-detector
```

2. Create and activate virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to:
```
http://localhost:10000
```

## Deployment on Render

### Method 1: Using Docker (Recommended)

1. Push your code to GitHub
2. Go to [Render Dashboard](https://dashboard.render.com/)
3. Click "New +" → "Web Service"
4. Connect your GitHub repository
5. Render will automatically detect the `Dockerfile`
6. Click "Create Web Service"

### Method 2: Using render.yaml (Blueprint)

1. Push your code to GitHub with the `render.yaml` file
2. Go to [Render Dashboard](https://dashboard.render.com/)
3. Click "New +" → "Blueprint"
4. Connect your GitHub repository
5. Render will automatically configure based on `render.yaml`

### Environment Variables

Set these in Render dashboard if needed:
- `FLASK_DEBUG`: Set to `false` for production

### Important Notes for Render Deployment

- Free tier may have cold starts (app sleeps after inactivity)
- Video processing might timeout on free tier - consider upgrading for production
- Model files (`best_anomaly_model.pt`) must be included in the repository
- Maximum request timeout: 120 seconds (configured in Dockerfile)

## File Structure

```
anomaly-detector/
├── app.py                      # Main Flask application
├── best_anomaly_model.pt       # YOLO model weights
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── Procfile                    # Gunicorn configuration
├── render.yaml                 # Render deployment config
├── runtime.txt                 # Python version
├── static/
│   ├── styles.css             # CSS styles
│   ├── uploads/               # Uploaded files (temporary)
│   └── processed/             # Processed results (temporary)
└── templates/
    ├── index.html             # Home page
    └── detector.html          # Detection page
```

## API Endpoints

### `GET /`
Home page

### `GET /detector`
Detection interface page

### `POST /detect`
Upload and process image/video

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: file (image or video)

**Response:**
```json
{
  "type": "image|video",
  "path": "/static/processed/filename"
}
```

## Supported File Formats

- **Images**: PNG, JPG, JPEG
- **Videos**: MP4, AVI, MOV

## Model Information

The application uses a custom-trained YOLOv8 model (`best_anomaly_model.pt`) for anomaly detection. Ensure this file is present in the root directory.

## Performance Optimization

- Using `opencv-python-headless` for serverless deployments
- NumPy version locked to <2 for PyTorch compatibility
- Gunicorn configured with:
  - 1 worker (for free tier memory limits)
  - 120s timeout (for video processing)

## Troubleshooting

### NumPy Compatibility Issues
If you encounter NumPy errors, ensure you're using NumPy version <2:
```bash
pip install "numpy<2"
```

### OpenCV Issues on Linux
Install system dependencies:
```bash
apt-get update && apt-get install -y ffmpeg libsm6 libxext6 libgl1-mesa-glx
```

## License

This project is open source and available under the MIT License.

## Author

Tarun - [GitHub](https://github.com/tarun-02005)
