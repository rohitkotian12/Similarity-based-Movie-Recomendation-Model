import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommender:
    def __init__(self, max_features: int = 5000):
        self.cv = CountVectorizer(max_features=max_features, stop_words='english')
        self.similarity_matrix = None
        self.df = None

    def fit(self, df: pd.DataFrame):
        """Fits the vectorizer and calculates the cosine similarity matrix."""
        self.df = df
        vector = self.cv.fit_transform(self.df['tags']).toarray()
        self.similarity_matrix = cosine_similarity(vector)

    def recommend(self, movie_title: str, top_n: int = 5) -> list:
        """Returns a list of recommended movie titles based on the input movie."""
        if self.similarity_matrix is None or self.df is None:
            raise ValueError("Model is not fitted. Call fit() with preprocessed data first.")
            
        try:
            # Find the index of the movie
            movie_idx = self.df[self.df['title'] == movie_title].index[0]
            
            # Calculate distances and sort
            distances = list(enumerate(self.similarity_matrix[movie_idx]))
            sorted_distances = sorted(distances, reverse=True, key=lambda x: x[1])
            
            # Fetch top N titles (skipping the first one, which is the movie itself)
            recommendations = []
            for i in sorted_distances[1:top_n+1]:
                recommendations.append(self.df.iloc[i[0]].title)
                
            return recommendations
            
        except IndexError:
            return [f"Movie '{movie_title}' not found in the dataset."]