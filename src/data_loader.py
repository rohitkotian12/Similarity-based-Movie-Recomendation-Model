import pandas as pd

class DataLoader:
    def __init__(self, movies_path: str, credits_path: str):
        self.movies_path = movies_path
        self.credits_path = credits_path

    def load_and_merge(self) -> pd.DataFrame:
        """Loads movies and credits data, then merges them on title."""
        try:
            movies = pd.read_csv(self.movies_path)
            credits = pd.read_csv(self.credits_path)
            # Merge and select required columns
            merged_df = movies.merge(credits, on='title')
            columns_to_keep = ['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']
            merged_df = merged_df[columns_to_keep].dropna()
            
            return merged_df
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            raise