# Quick Deployment Guide

## 🚀 Deploy to Render in 5 Steps

### Step 1: Initialize Git (if needed)
```powershell
git init
```

### Step 2: Add Files to Git
```powershell
git add .
```

### Step 3: Commit Changes
```powershell
git commit -m "Initial commit - Anomaly Detector ready for deployment"
```

### Step 4: Push to GitHub
```powershell
# Add your GitHub repository
git remote add origin https://github.com/tarun-02005/anomaly-detector.git

# Push to main branch
git push -u origin main
```

### Step 5: Deploy on Render

1. **Go to Render**: https://dashboard.render.com/
2. **Click**: "New +" → "Web Service"
3. **Connect**: Your GitHub repository (`tarun-02005/anomaly-detector`)
4. **Configure**:
   - Name: `anomaly-detector`
   - Environment: `Docker`
   - Region: Choose closest to you
   - Branch: `main`
   - Plan: `Free`
5. **Click**: "Create Web Service"

### Step 6: Wait & Test

- Build time: 5-10 minutes
- Once deployed, you'll get a URL like: `https://anomaly-detector.onrender.com`
- Test by uploading an image!

---

## 📝 Alternative: One-Click Render Blueprint

If you pushed `render.yaml` to your repo:

1. Go to: https://dashboard.render.com/
2. Click: "New +" → "Blueprint"
3. Connect repository
4. Click: "Apply"

Render will automatically configure everything! ✨

---

## 🔧 Git Commands Reference

### Update after changes:
```powershell
git add .
git commit -m "Update description"
git push
```

Render will auto-deploy on every push! 🎉

---

## ⚠️ Important Notes

1. **Model file**: Make sure `best_anomaly_model.pt` is committed
2. **File size**: GitHub has 100MB limit per file
3. **First request**: May take 30s on free tier (cold start)
4. **Videos**: Large videos may timeout on free tier

---

## 📊 Monitoring Your Deployment

### Render Dashboard:
- **Logs**: View build and runtime logs
- **Events**: Track deployments
- **Metrics**: Monitor CPU/Memory
- **Shell**: Access container shell if needed

### Check Logs:
- Build logs: See installation progress
- Runtime logs: Monitor requests and errors

---

## 🎯 What's Deployed

✅ Files included:
- `app.py` - Main application
- `best_anomaly_model.pt` - YOLO model
- `templates/` - HTML files
- `static/` - CSS and assets
- All deployment configs

✅ Configuration:
- Python 3.12
- Gunicorn WSGI server
- 120s timeout
- Auto-scaling enabled
- Health checks configured

---

## 🐛 Troubleshooting

### Build fails?
```powershell
# Check requirements.txt is correct
cat requirements.txt

# Verify model file exists
ls -la best_anomaly_model.pt
```

### App not responding?
- Check Render logs
- Verify FLASK_DEBUG=false
- Check health endpoint: `/`

### Want to test locally first?
```powershell
.\venv\Scripts\python.exe test_deployment.py
```

---

## 🎉 Success!

Your app is live! Share your deployment:
- URL: `https://your-app-name.onrender.com`
- GitHub: `https://github.com/tarun-02005/anomaly-detector`

Happy detecting! 🔍✨
