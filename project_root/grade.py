# grade.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Import the pre-loaded global model
from model_loader import MODEL 
from ocr import run_ocr


def grade_submission(notes_file, answer_sheet_file, question_prompt=""):
    """
    Grades a student submission by comparing it against study materials and generating feedback.
    
    Args:
        notes_file: FileStorage object containing study material/teacher notes
        answer_sheet_file: FileStorage object containing student's answer sheet
        question_prompt: The specific exam question/prompt
        
    Returns:
        Dictionary with grade, feedback, and other relevant information
    """

    # 1. Run OCR on the Answer Sheet
    ocr_results = run_ocr(answer_sheet_file)
    
    # Check if OCR returned an error (dictionary with 'error' key)
    if isinstance(ocr_results, dict) and "error" in ocr_results:
        return {"error": f"Failed to process answer sheet: {ocr_results['error']}"}

    # 2. Extract text from notes file
    notes_ocr_results = run_ocr(notes_file)
    if isinstance(notes_ocr_results, dict) and "error" in notes_ocr_results:
        return {"error": f"Failed to process notes file: {notes_ocr_results['error']}"}

    # 3. Extract Relevant Information
    student_answer = ocr_results
    model_answer = notes_ocr_results  # Use the notes as the reference answer

    # 4. Calculate Semantic Score
    score_result = calculate_semantic_score(model_answer, student_answer, max_marks=10.0)

    # 5. Generate detailed feedback using Gemini AI
    feedback = generate_detailed_feedback(question_prompt, student_answer, model_answer, score_result)

    # 6. Return results in the format expected by the template
    return {
        "grade": f"{score_result['score']}/10.0",
        "feedback": feedback,
        "student_answer": student_answer,
        "model_answer": model_answer,
        "score_details": score_result
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

def generate_detailed_feedback(question_prompt, student_answer, model_answer, score_result):
    """
    Generates detailed feedback using Gemini AI based on the student's answer, model answer, and scoring.
    
    Args:
        question_prompt: The exam question/prompt
        student_answer: The student's extracted answer
        model_answer: The reference answer from notes
        score_result: The scoring results dictionary
        
    Returns:
        String containing detailed feedback
    """
    try:
        from ocr import client
        
        if client is None:
            # Fallback feedback if Gemini is not available
            return f"""
SCORE: {score_result['score']}/10.0 ({score_result['similarity']*100:.1f}% similarity)

ANALYSIS:
- Your answer was compared against the reference material
- Semantic similarity: {score_result['similarity']*100:.1f}%
- {score_result['reason']}

STUDENT ANSWER:
{student_answer}

Please set up the GEMINI_API_KEY environment variable to get detailed AI-powered feedback.
"""

        # Create a comprehensive prompt for feedback generation
        feedback_prompt = f"""
You are an expert educational assessor. Please provide detailed feedback for a student's exam answer.

QUESTION: {question_prompt}

REFERENCE ANSWER (from study materials):
{model_answer}

STUDENT ANSWER:
{student_answer}

SCORE: {score_result['score']}/10.0 (Semantic Similarity: {score_result['similarity']*100:.1f}%)

Please provide:
1. A brief summary of what the student got right
2. Areas where the student could improve
3. Specific suggestions for better understanding
4. Key concepts that were missed or misunderstood
5. Encouragement and constructive guidance

Format your response in a clear, educational manner suitable for student learning.
"""

        # Call Gemini API for feedback generation
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[feedback_prompt],
            config={"system_instruction": "You are a helpful educational assistant focused on providing constructive feedback to help students learn and improve."}
        )
        
        # Extract the feedback text
        if hasattr(response, 'text'):
            feedback_text = response.text
        elif hasattr(response, 'candidates') and response.candidates:
            feedback_text = response.candidates[0].content.parts[0].text
        else:
            feedback_text = str(response)
            
        return feedback_text
        
    except Exception as e:
        # Fallback feedback if Gemini call fails
        return f"""
SCORE: {score_result['score']}/10.0 ({score_result['similarity']*100:.1f}% similarity)

ANALYSIS:
- Your answer was compared against the reference material
- Semantic similarity: {score_result['similarity']*100:.1f}%
- {score_result['reason']}

STUDENT ANSWER:
{student_answer}

REFERENCE ANSWER:
{model_answer}

Note: Detailed AI feedback is temporarily unavailable. Please try again later.
Error: {str(e)}
"""

# NOTE: You can add other simple grading functions here (e.g., for MCQ/Keywords) 
# but this is your main semantic scoring function.