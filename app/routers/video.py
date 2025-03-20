import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
from io import BytesIO
from transformers import pipeline

router = APIRouter(
    prefix="/api/video",
    tags=["video"],
)

# Initialize the facial expression recognition pipeline
try:
    expression_pipeline = pipeline("image-classification", model="motheecreator/vit-Facial-Expression-Recognition")
except Exception as e:
    print(f"Error initializing model: {str(e)}")
    expression_pipeline = None

@router.post("/expression")
async def detect_facial_expression(
    image_file: UploadFile = File(...),
):
    """
    Detect facial expression in an image:
    1. Load the image
    2. Run it through the facial expression recognition model
    3. Return the classification results
    
    Args:
        image_file: Uploaded image file
        
    Returns:
        JSON response with facial expression predictions
    """
    if not image_file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if expression_pipeline is None:
        raise HTTPException(status_code=500, detail="Facial expression model not initialized")
    
    try:
        # Read the image file
        contents = await image_file.read()
        
        # Convert to PIL Image for the transformers pipeline
        image = Image.open(BytesIO(contents))
        
        # Run the image through the pipeline
        result = expression_pipeline(image)
        
        return JSONResponse(
            content={
                "message": "Image processed successfully",
                "predictions": result
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


