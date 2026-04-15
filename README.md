A full-stack Machine Learning application that recommends movies based on content similarity. The project features a FastAPI backend, a Natural Language Processing (NLP) pipeline for data preprocessing, and is containerized using Docker.

Features:
Content-Based Filtering: Recommends movies based on genres, keywords, cast, and crew.
NLP Pipeline: Implements stemming and vectorization to process movie overviews and metadata.
FastAPI Backend: High-performance API endpoints for fetching recommendations and checking system health.
CI/CD Integrated: Includes GitHub Actions for automated build and execution testing.
Dockerized: Ready for deployment in any environment.

Project Structure:
├── data/                   # Dataset files (CSV)
├── src/
│   ├── data_loader.py      # Handles data ingestion and merging 
│   ├── preprocessor.py     # Cleans data and generates tags 
│   └── model.py            # Cosine similarity-based recommender engine 
├── static/                 # Web UI (index.html)
├── main.py                 # FastAPI application and entry point 
├── evaluate.py             # Accuracy and performance evaluation 
├── Dockerfile              # Containerization instructions 
├── requirements.txt        # Python dependencies [cite: 2]
└── .github/workflows/      # mlops.yaml for CI/CD

Installation & Setup:
1. Clone the repository-
  git clone [https://github.com/your-username/movie-recommender.git
  cd movie-recommender](https://github.com/rohitkotian12/Similarity-based-Movie-Recomendation-Model/tree/main?tab=readme-ov-file)

2. Local Environment Setup-
Ensure you have Python 3.12 installed.
# Create virtual environment
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
# Install dependencies
  pip install -r requirements.txt

3. Run the Application-
   python main.py

Docker Deployment:
You can run the entire application using Docker
# Build the image
  docker build -t movie-recommender . 
# Run the container
  docker run -p 8000:8000 movie-recommender

Machine Learning Pipeline:
Data Loading: Merges TMDB movies and credits datasets on titles.
Preprocessing:Extracts genres, keywords, top 3 cast members, and directors.
Applies PorterStemmer to reduce words to their root forms.
Combines metadata into a single tags column.
Vectorization: Uses CountVectorizer to convert text tags into vectors.
Similarity: Calculates Cosine Similarity between movie vectors to find the closest matches.

Testing & Evaluation:
To verify the model's training consistency
  python evaluate.py
This script checks if the model can correctly identify a movie as its own closest match within the similarity matrix.

CI/CD:
The project uses GitHub Actions (mlops.yaml) to automatically:
Set up a Python environment.
Install dependencies.
Run the main application to ensure code integrity on every push.
