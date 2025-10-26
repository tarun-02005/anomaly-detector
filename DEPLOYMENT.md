# Render Deployment Checklist

## ✅ Pre-Deployment Checklist

### 1. Files Ready
- [x] `app.py` - Main application file
- [x] `requirements.txt` - All dependencies listed
- [x] `Dockerfile` - Docker configuration
- [x] `Procfile` - Gunicorn configuration
- [x] `render.yaml` - Render blueprint
- [x] `runtime.txt` - Python version specified
- [x] `.dockerignore` - Excludes unnecessary files
- [x] `.gitignore` - Git ignore rules
- [x] `README.md` - Documentation

### 2. Model Files
- [x] `best_anomaly_model.pt` - YOLO model weights (must be in repo)
- [ ] Optional: `best_anomaly_model.onnx` (if using ONNX)

### 3. Code Updates
- [x] Port configured via environment variable
- [x] Debug mode controlled via FLASK_DEBUG env var
- [x] opencv-python-headless for serverless
- [x] NumPy version locked to <2
- [x] Gunicorn installed and configured
- [x] Directories created in Dockerfile

## 📋 Deployment Steps

### Step 1: Push to GitHub

```bash
# Initialize git (if not already)
git init

# Add all files
git add .

# Commit changes
git commit -m "Prepare for Render deployment"

# Add remote repository
git remote add origin https://github.com/tarun-02005/anomaly-detector.git

# Push to GitHub
git push -u origin main
```

### Step 2: Deploy on Render

#### Option A: Docker Deployment (Recommended)
1. Go to https://dashboard.render.com/
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: anomaly-detector
   - **Environment**: Docker
   - **Plan**: Free
   - **Dockerfile Path**: ./Dockerfile
5. Click "Create Web Service"

#### Option B: Blueprint Deployment
1. Go to https://dashboard.render.com/
2. Click "New +" → "Blueprint"
3. Connect your GitHub repository
4. Render will read `render.yaml` automatically
5. Click "Apply"

### Step 3: Verify Deployment

1. Wait for build to complete (5-10 minutes)
2. Click on the service URL
3. Test image upload
4. Test video upload (may be slower on free tier)

## ⚙️ Configuration

### Environment Variables (Optional)
Set in Render Dashboard → Environment:
- `FLASK_DEBUG` = `false` (already in render.yaml)
- `PORT` = Auto-set by Render

### Important Settings
- **Region**: Choose closest to users
- **Auto-Deploy**: Enabled (deploys on git push)
- **Health Check Path**: `/`

## 🔍 Troubleshooting

### Build Fails
- Check Render logs for errors
- Verify all dependencies are in requirements.txt
- Ensure model file is committed to repo

### Timeout Issues
- Increase timeout in Dockerfile (already set to 120s)
- For large videos, consider upgrading from free tier
- Free tier may sleep after 15 min inactivity

### Memory Issues
- Reduce workers to 1 (already configured)
- Free tier has 512MB RAM limit
- Consider upgrading for production

### Cold Starts
- Free tier apps sleep after inactivity
- First request after sleep takes 30-60 seconds
- Consider paid plan for always-on service

## 📊 Monitoring

After deployment, monitor:
- Build logs: Check for errors
- Runtime logs: Monitor requests
- Metrics: CPU, Memory usage
- Errors: Track failed requests

## 🚀 Post-Deployment

### Test Your App
```bash
# Replace with your Render URL
curl https://anomaly-detector.onrender.com/

# Test with image upload
curl -X POST -F "file=@test_image.jpg" https://anomaly-detector.onrender.com/detect
```

### Keep App Awake (Free Tier)
- Use a service like UptimeRobot to ping your app every 5-10 minutes
- URL to ping: `https://your-app.onrender.com/`

## 📝 Notes

- First deployment takes 5-10 minutes
- Free tier limitations:
  - 512 MB RAM
  - Sleeps after 15 min inactivity
  - 750 hours/month free
- Model file size impacts build time
- Video processing may timeout on large files (free tier)

## 🎉 Success!

Your app is deployed at: `https://anomaly-detector.onrender.com`

Happy detecting! 🔍
