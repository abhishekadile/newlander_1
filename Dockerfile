# Use Node.js 18 on Debian Bullseye
FROM node:20-bullseye

# Install system dependencies
# - opencfu: The core colony counting tool
# - python3, python3-pip: For preprocessing script
# - libgl1: Required for OpenCV (cv2)
RUN apt-get update && apt-get install -y \
    opencfu \
    python3 \
    python3-pip \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies for preprocess.py
RUN pip3 install --no-cache-dir \
    numpy \
    opencv-python-headless

# Set environment variable for Python binary
ENV PYTHON_BIN=python3
# Set environment variable for OpenCFU binary (optional, as code will default to 'opencfu')
ENV OPENCFU_BIN=opencfu

# Copy package files first for caching
COPY package*.json ./

# Install Node.js dependencies
RUN npm install

# Copy the rest of the application code
# CRITICAL: We explicitly copy core_engine to preserve directory structure
# even if we use the system-installed opencfu binary.
COPY . .

# Expose the application port
EXPOSE 3000

# Start the server
CMD ["node", "server.js"]
