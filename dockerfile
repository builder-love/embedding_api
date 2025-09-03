# Use your pre-built golden image from Artifact Registry
FROM us-central1-docker.pkg.dev/data-pipelines-450611/embedding-services/qwen-embedding-base:0.1.0

# Set the working directory (it's already /app, but good practice to state it)
WORKDIR /app

# The model and requirements are already installed.
# We only need to copy the application code
COPY main.py .

# Expose the port the app runs on
EXPOSE 8080

# Define the command to run your app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]