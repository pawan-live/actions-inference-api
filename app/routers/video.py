import tempfile
import os
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List
from pydantic import BaseModel, HttpUrl

from app.services.video_processing import extract_frames, extract_middle_frame, save_landmarks_visualization
from app.services.face_detection import detect_face_landmarks
from app.utils.helpers import download_video_from_url

router = APIRouter(
    prefix="/api/video",
    tags=["video"],
)

@router.post("/process")
async def process_video(
    background_tasks: BackgroundTasks,
    video_file: UploadFile = File(...),
):
    """
    Process a video file:
    1. Extract frames from video
    2. Detect face landmarks in each frame
    3. Process landmarks
    4. Return results
    """
    if not video_file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.filename)[1]) as temp_file:
            temp_file.write(await video_file.read())
            temp_path = temp_file.name
        
        # Process the video
        frames = extract_frames(temp_path)
        if not frames:
            raise HTTPException(status_code=400, detail="Could not extract frames from video")
            
        # Process each frame with MediaPipe
        results = []
        for i, frame in enumerate(frames):
            landmarks = detect_face_landmarks(frame)
            if landmarks:
                results.append({
                    "frame_number": i,
                    "landmarks": landmarks
                })
        
        # Clean up the temporary file
        background_tasks.add_task(os.unlink, temp_path)
        
        return JSONResponse(
            content={
                "message": "Video processed successfully",
                "total_frames": len(frames),
                "frames_with_faces": len(results),
                "results": results[:5]  # Return only first 5 frames as example
            }
        )
        
    except Exception as e:
        # Make sure to clean up if there's an error
        if 'temp_path' in locals():
            os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

class VideoURL(BaseModel):
    url: HttpUrl

@router.post("/process-url")
async def process_video_url(
    background_tasks: BackgroundTasks,
    video_url: VideoURL,
):
    """
    Process a video from a URL:
    1. Download video from URL
    2. Extract frames from video
    3. Detect face landmarks in each frame
    4. Process landmarks
    5. Return results
    """
    try:
        # Download the video from the URL
        temp_path = download_video_from_url(str(video_url.url))
        
        # Process the video
        frames = extract_frames(temp_path)
        if not frames:
            raise HTTPException(status_code=400, detail="Could not extract frames from video")
            
        # Process each frame with MediaPipe
        results = []
        for i, frame in enumerate(frames):
            landmarks = detect_face_landmarks(frame)
            if landmarks:
                results.append({
                    "frame_number": i,
                    "landmarks": landmarks
                })
        
        # Clean up the temporary file
        background_tasks.add_task(os.unlink, temp_path)
        
        return JSONResponse(
            content={
                "message": "Video processed successfully",
                "total_frames": len(frames),
                "frames_with_faces": len(results),
                "results": results[:5]  # Return only first 5 frames as example
            }
        )
        
    except Exception as e:
        # Make sure to clean up if there's an error
        if 'temp_path' in locals():
            os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

@router.post("/process-url-middle-frame")
async def process_video_url_middle_frame(
    background_tasks: BackgroundTasks,
    video_url: VideoURL,
):
    """
    Process the middle frame of a video from a URL:
    1. Download video from URL
    2. Extract the middle frame from the video
    3. Detect face landmarks in the frame
    4. Save visualization of landmarks
    5. Return landmark coordinates and visualization path
    """
    try:
        # Download the video from the URL
        temp_path = download_video_from_url(str(video_url.url))
        
        # Extract the middle frame
        middle_frame = extract_middle_frame(temp_path)
        if middle_frame is None:
            raise HTTPException(status_code=400, detail="Could not extract middle frame from video")
            
        # Process the frame with MediaPipe
        landmarks = detect_face_landmarks(middle_frame)
        if not landmarks:
            raise HTTPException(status_code=400, detail="No face detected in the middle frame")
        
        # Save visualization of landmarks
        vis_path = save_landmarks_visualization(middle_frame, landmarks)
        
        # Clean up the temporary file
        background_tasks.add_task(os.unlink, temp_path)
        
        return JSONResponse(
            content={
                "message": "Middle frame processed successfully",
                "landmarks": landmarks,
                "visualization_path": vis_path
            }
        )
        
    except Exception as e:
        # Make sure to clean up if there's an error
        if 'temp_path' in locals():
            os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
