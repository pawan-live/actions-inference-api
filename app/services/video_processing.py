import cv2
import numpy as np
import os
import time
from typing import List, Optional

def extract_frames(video_path: str, max_frames: Optional[int] = None, sample_rate: int = 1) -> List[np.ndarray]:
    """
    Extract frames from a video file
    
    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to extract (None = all frames)
        sample_rate: Extract every nth frame
        
    Returns:
        List of frames as numpy arrays
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % sample_rate == 0:
            frames.append(frame)
            
        frame_count += 1
        
        if max_frames and len(frames) >= max_frames:
            break
    
    cap.release()
    return frames

def extract_middle_frame(video_path: str) -> Optional[np.ndarray]:
    """
    Extract the middle frame from a video file
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Middle frame as numpy array or None if extraction fails
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None
    
    # Calculate middle frame index
    middle_frame_idx = total_frames // 2
    
    # Set the frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
    
    # Read the frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    
    return frame

def save_landmarks_visualization(frame: np.ndarray, landmarks_list, output_dir: str = "./output") -> str:
    """
    Save a visualization of facial landmarks on a frame
    
    Args:
        frame: The frame to visualize landmarks on
        landmarks_list: List of landmarks to visualize
        output_dir: Directory to save the visualization
        
    Returns:
        Path to the saved visualization
    """
    from app.services.face_detection import visualize_landmarks
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize landmarks on the frame
    vis_frame = visualize_landmarks(frame, landmarks_list)
    
    # Generate a unique filename with timestamp
    timestamp = int(time.time())
    filename = f"landmarks_{timestamp}.jpg"
    filepath = os.path.join(output_dir, filename)
    
    # Save the visualization
    cv2.imwrite(filepath, vis_frame)
    
    return filepath
