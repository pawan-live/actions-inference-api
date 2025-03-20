import mediapipe as mp
import numpy as np
import cv2
from typing import List, Dict, Any, Optional

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

def detect_face_landmarks(frame: np.ndarray) -> Optional[List[Dict[str, Any]]]:
    """
    Detect face landmarks in a frame using MediaPipe
    
    Args:
        frame: Image as numpy array
        
    Returns:
        List of landmarks or None if no face detected
    """
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image and get face landmarks
    results = face_mesh.process(rgb_frame)
    
    if not results.multi_face_landmarks:
        return None
    
    landmarks_list = []
    
    for face_landmarks in results.multi_face_landmarks:
        landmarks = []
        for idx, landmark in enumerate(face_landmarks.landmark):
            landmarks.append({
                "index": idx,
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z
            })
        landmarks_list.append(landmarks)
    
    return landmarks_list

def visualize_landmarks(frame: np.ndarray, landmarks_list: List) -> np.ndarray:
    """
    Visualize the landmarks on the frame
    
    Args:
        frame: Original image
        landmarks_list: List of landmarks
        
    Returns:
        Frame with visualized landmarks
    """
    vis_frame = frame.copy()
    
    # Draw points for each landmark
    for face_landmarks in landmarks_list:
        # Convert landmarks to pixel coordinates
        h, w, _ = vis_frame.shape
        points = []
        for landmark in face_landmarks:
            x = int(landmark["x"] * w)
            y = int(landmark["y"] * h)
            # Draw circle at each landmark point
            cv2.circle(vis_frame, (x, y), 1, (0, 255, 0), -1)
            points.append((x, y))
        
        # Draw connections between adjacent points to form a mesh (simplified)
        # This is a basic visualization - you can make it more detailed if needed
        for i in range(len(points) - 1):
            cv2.line(vis_frame, points[i], points[i + 1], (0, 255, 0), 1)
    
    return vis_frame

def determine_face_angle(landmarks: List[Dict[str, Any]]) -> str:
    """
    Determine if the face is looking at the screen or away based on nose and cheek positions
    
    Args:
        landmarks: List of face landmarks
        
    Returns:
        String indicating face direction: "looking_at_screen", "looking_left", "looking_right"
    """
    # Find the required landmarks
    nose_tip = None
    right_cheek = None
    left_cheek = None
    
    for landmark in landmarks:
        if landmark["index"] == 1:
            nose_tip = landmark
        elif landmark["index"] == 234:
            right_cheek = landmark
        elif landmark["index"] == 454:
            left_cheek = landmark
    
    # If we couldn't find all required landmarks
    if not (nose_tip and right_cheek and left_cheek):
        return "unknown"
    
    # Calculate distances from nose tip to cheeks (using x-coordinates)
    distance_to_right = abs(nose_tip["x"] - right_cheek["x"])
    distance_to_left = abs(nose_tip["x"] - left_cheek["x"])
    
    # Determine face direction
    threshold = 0.05  # Threshold for considering the face looking straight
    
    if abs(distance_to_right - distance_to_left) < threshold:
        return "looking_at_screen"
    elif distance_to_right < distance_to_left:
        return "looking_right"  # Nose is closer to right cheek
    else:
        return "looking_left"   # Nose is closer to left cheek
