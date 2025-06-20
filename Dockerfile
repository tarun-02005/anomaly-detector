# Use official Python 3.12 base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy dependencies list
COPY requirements.txt .

# Install OS-level dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy project files into container
COPY . .

# Expose the port Flask runs on
EXPOSE 10000

# Run the Flask app using gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]
