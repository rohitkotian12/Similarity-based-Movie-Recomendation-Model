import ast
import pandas as pd
import nltk
from nltk.stem import PorterStemmer

class DataPreprocessor:
    def __init__(self):
        self.ps = PorterStemmer()

    @staticmethod
    def extract_names(text: str) -> list:
        """Extracts the 'name' key from a stringified list of dictionaries."""
        return [i['name'] for i in ast.literal_eval(text)]

    @staticmethod
    def extract_top_3_cast(text: str) -> list:
        """Extracts top 3 cast members."""
        return [i['name'] for i in ast.literal_eval(text)][:3]

    @staticmethod
    def fetch_director(text: str) -> list:
        """Extracts the director's name."""
        for i in ast.literal_eval(text):
            if i['job'] == 'Director':
                return [i['name']]
        return []

    @staticmethod
    def remove_spaces(word_list: list) -> list:
        """Removes spaces from words to create unique tags."""
        return [i.replace(" ", "") for i in word_list]

    def stem_text(self, text: str) -> str:
        """Stems a string of text using PorterStemmer."""
        return " ".join([self.ps.stem(word) for word in text.split()])

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Runs the full preprocessing pipeline."""
        df = df.copy()
        
        # Parse strings to lists
        df['genres'] = df['genres'].apply(self.extract_names)
        df['keywords'] = df['keywords'].apply(self.extract_names)
        df['cast'] = df['cast'].apply(self.extract_top_3_cast)
        df['crew'] = df['crew'].apply(self.fetch_director)
        df['overview'] = df['overview'].apply(lambda x: x.split())

        # Remove spaces
        for col in ['genres', 'keywords', 'cast', 'crew']:
            df[col] = df[col].apply(self.remove_spaces)

        # Concatenate features into tags
        df['tags'] = df['overview'] + df['genres'] + df['keywords'] + df['cast'] + df['crew']
        
        # Create final dataframe
        new_df = df[['movie_id', 'title', 'tags']].copy()
        new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())
        new_df['tags'] = new_df['tags'].apply(self.stem_text)

        return new_df