FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for TensorFlow + OpenCV)
RUN apt-get update && apt-get install -y \
    python3-opencv \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Expose port
EXPOSE 7860

# Run Flask
CMD ["python", "app.py"]