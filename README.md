# Expression Prediction API

A FastAPI application that processes videos to detect and analyze facial expressions using MediaPipe.

## Setup

1. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Run the server:

   ```
   uvicorn app.main:app --reload
   ```

3. Access the API documentation at http://localhost:8000/docs

## API Endpoints

- `POST /api/video/process`: Upload a video file for face landmark detection
- `POST /api/video/process-url`: Provide a video URL for face landmark detection

## Requirements

- Python 3.8+
- OpenCV
- MediaPipe
- FastAPI

## Running with Docker

### Build and run using Docker

1. Build the Docker image:

   ```
   docker build -t expression-prediction-api .
   ```

2. Run the container:
   ```
   docker run -p 8000:8000 expression-prediction-api
   ```

### Or using Docker Compose

1. Build and run:

   ```
   docker-compose up
   ```

2. Access the API documentation at http://localhost:8000/docs
