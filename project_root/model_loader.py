# model_loader.py

from sentence_transformers import SentenceTransformer

# This model will be loaded once when the application starts
try:
    # GLOBAL VARIABLE: The model object
    MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    print("Semantic scoring model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}. Semantic scoring disabled.")
    MODEL = None