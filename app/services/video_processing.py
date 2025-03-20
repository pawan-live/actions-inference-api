import cv2
import numpy as np
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
