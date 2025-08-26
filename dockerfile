# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies that might be needed by torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# --- Pre-download the correct model into the image ---
RUN python -c "from transformers import AutoModel, AutoTokenizer; \
               MODEL_NAME = 'Qwen/Qwen3-Embedding-4B'; \
               AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True); \
               AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)"

# Copy the application code into the container
COPY main.py .

# Expose the port the app runs on
EXPOSE 8080

# Define the command to run your app using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]