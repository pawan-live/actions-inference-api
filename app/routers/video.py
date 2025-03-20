import tempfile
import os
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List

from app.services.video_processing import extract_frames
from app.services.face_detection import detect_face_landmarks

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
