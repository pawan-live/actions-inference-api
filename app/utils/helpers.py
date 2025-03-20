import time
import logging
import tempfile
import requests
import os
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def timing_decorator(func):
    """Decorator to measure the execution time of functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Function {func.__name__} took {end_time - start_time:.2f} seconds to run")
        return result
    return wrapper

def download_video_from_url(url: str) -> str:
    """Download a video from a URL to a temporary file
    
    Args:
        url: URL of the video
        
    Returns:
        Path to the downloaded video file
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        
        # Get file extension from URL or default to .mp4
        file_extension = os.path.splitext(url.split('/')[-1])[-1]
        if not file_extension:
            file_extension = '.mp4'
            
        # Save to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        for chunk in response.iter_content(chunk_size=8192):
            temp_file.write(chunk)
        
        temp_file.close()
        logger.info(f"Video downloaded from URL to {temp_file.name}")
        return temp_file.name
    except Exception as e:
        logger.error(f"Error downloading video from URL: {e}")
        raise
