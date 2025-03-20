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
    
    for face_landmarks in landmarks_list:
        # Create a MediaPipe FaceLandmark object
        mp_landmarks = mp_face_mesh.FaceLandmark()
        for landmark in face_landmarks:
            mp_landmarks.landmark.append(
                mp.framework.formats.landmark_pb2.NormalizedLandmark(
                    x=landmark["x"],
                    y=landmark["y"],
                    z=landmark["z"]
                )
            )
        
        # Draw the landmarks
        mp_drawing.draw_landmarks(
            image=vis_frame,
            landmark_list=mp_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
        )
    
    return vis_frame
