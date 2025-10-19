# grade.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Import the pre-loaded global model
from model_loader import MODEL 
from ocr import run_ocr


def grade_submission(notes_file, answer_sheet_file):

    # 1. Run OCR on the Answer Sheet
    ocr_results = run_ocr(answer_sheet_file)
    if "error" in ocr_results:
        return {"error": f"Failed to process answer sheet: {ocr_results['error']}"}

    # 2. Extract Relevant Information
    student_answer = ocr_results
    model_answer = "Expected answer from notes"  # Placeholder for actual model answer extraction

    # 3. Calculate Semantic Score
    score = calculate_semantic_score(model_answer, student_answer, max_marks=10.0)

    return {
        "student_answer": student_answer,
        "model_answer": model_answer,
        "score": score
    }

def calculate_semantic_score(model_answer: str, student_answer: str, max_marks: float) -> dict:
    """
    Calculates a score for a subjective answer by comparing its semantic meaning
    to the model answer using Cosine Similarity on Sentence Embeddings.
    
    Returns a dictionary with the score, raw similarity, and a reason.
    """
    
    # 1. Check for Model Availability and Empty Answer
    if MODEL is None:
        return {"score": 0.0, "similarity": 0.0, "reason": "AI model failed to load. Cannot grade semantically."}

    if not student_answer or not student_answer.strip():
        return {"score": 0.0, "similarity": 0.0, "reason": "Student answer was empty or only whitespace."}
    
    try:
        # 2. Generate Embeddings
        # The model converts the text into numerical vectors (embeddings)
        embeddings = MODEL.encode([model_answer, student_answer])
        
        # 3. Calculate Cosine Similarity (a value between 0.0 and 1.0)
        similarity_score = cosine_similarity(
            embeddings[0].reshape(1, -1),
            embeddings[1].reshape(1, -1)
        )[0][0]
        
        # 4. Calculate Final Score
        # Scale the 0-1 similarity score to the question's max_marks
        assigned_marks = round(float(similarity_score) * max_marks, 2)
        
        return {
            "score": assigned_marks,
            "similarity": float(similarity_score),
            "reason": f"Scored based on {round(float(similarity_score)*100, 1)}% semantic similarity."
        }
        
    except Exception as e:
        # Fallback for unexpected errors during encoding/similarity
        return {"score": 0.0, "similarity": 0.0, "reason": f"Error during AI grading: {e}"}

# NOTE: You can add other simple grading functions here (e.g., for MCQ/Keywords) 
# but this is your main semantic scoring function.