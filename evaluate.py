import numpy as np

from src.model import MovieRecommender
from src.data_loader import DataLoader
from src.preprocessor import DataPreprocessor

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

# Compute training accuracy as self-consistency on the fitted similarity matrix.
# For each movie, the most similar item should be itself if the model is trained correctly.
similarity_matrix = recommender.similarity_matrix
if similarity_matrix is not None:
    self_matches = np.argmax(similarity_matrix, axis=1)
    training_accuracy = np.mean(self_matches == np.arange(similarity_matrix.shape[0]))
    print(f"Training accuracy: {training_accuracy * 100:.2f}%")
else:
    print("Training accuracy could not be computed: similarity matrix is missing.")


