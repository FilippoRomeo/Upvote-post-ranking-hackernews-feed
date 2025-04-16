# Use official PyTorch image with CUDA or CPU depending on your machine
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY src/ ./src/

# Set working directory to src for script execution
WORKDIR /app/src

# Run training script by default
CMD ["python", "tokenizer.py"]
