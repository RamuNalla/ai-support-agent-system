# Use a slim Python 3.10 image as the base, now using Bookworm (Debian 12)
FROM python:3.10-slim-bookworm

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install Python dependencies
# This step is done separately to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the pre-built FAISS index file into the Docker image.
RUN mkdir -p agent_service/
COPY agent_service/faiss_index.bin agent_service/faiss_index.bin

# Copy the entire application code into the container
# This copies the 'app' directory and any other files in agent_service
COPY app /app/app

# Expose port 8000 for the FastAPI application
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
