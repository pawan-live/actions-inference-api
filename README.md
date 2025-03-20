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

## Requirements

- Python 3.8+
- OpenCV
- MediaPipe
- FastAPI
