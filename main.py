from fastapi import FastAPI, staticfiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import os

from src.data_loader import DataLoader
from src.preprocessor import DataPreprocessor
from src.model import MovieRecommender

# Initialize FastAPI app
app = FastAPI(title="Movie Recommender")

# Global model instance
recommender = None

# Define request/response models
class PredictionRequest(BaseModel):
    movie_name: str

class PredictionResponse(BaseModel):
    movie_name: str
    recommendations: list
    found: bool


def initialize_model():
    """Initialize the machine learning model."""
    global recommender
    
    MOVIES_PATH = "data/tmdb_5000_movies.csv"
    CREDITS_PATH = "data/tmdb_5000_credits.csv"

    print("Loading data...")
    loader = DataLoader(MOVIES_PATH, CREDITS_PATH)
    raw_df = loader.load_and_merge()

    print("Preprocessing data...")
    preprocessor = DataPreprocessor()
    processed_df = preprocessor.preprocess(raw_df)

    print("Training model...")
    recommender = MovieRecommender(max_features=5000)
    recommender.fit(processed_df)
    print("Model ready!")


# Serve static files
if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", staticfiles.StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    """Serve the main HTML page."""
    return FileResponse("static/index.html")


@app.post("/predict")
async def predict(request: PredictionRequest):
    """Get movie recommendations based on input movie name."""
    if recommender is None:
        return {"error": "Model not initialized"}
    
    movie_name = request.movie_name.strip()
    
    if not movie_name:
        return {"error": "Movie name cannot be empty"}
    
    recommendations = recommender.recommend(movie_name, top_n=5)
    
    # Check if movie was found (if response contains "not found" message)
    found = not any("not found" in str(rec).lower() for rec in recommendations)
    
    return PredictionResponse(
        movie_name=movie_name,
        recommendations=recommendations,
        found=found
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": recommender is not None}


def main():
    """Initialize model and start the server."""
    initialize_model()
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )


if __name__ == "__main__":
    main()