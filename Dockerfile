# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


# Copy the backend requirements file into the container at /app/backend
COPY backend/requirements.txt /app/backend/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy the rest of the application code
COPY backend /app/backend
COPY frontend /app/frontend
COPY run.py /app/run.py

# Create necessary directories for volumes
RUN mkdir -p /app/model /app/data /app/uploads /app/outputs

# Expose port 8000
EXPOSE 8000

# Define environment variables
ENV PYTHONUNBUFFERED=1
ENV PIXELBLEND_HOST=0.0.0.0
ENV PIXELBLEND_PORT=8000

# Run the application
CMD ["python3", "run.py"]
