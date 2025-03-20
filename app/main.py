from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import video

app = FastAPI(
    title="Expression Prediction API",
    description="API for facial expression prediction using MediaPipe",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(video.router)

@app.get("/")
async def root():
    return {"message": "Welcome to the Expression Prediction API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
