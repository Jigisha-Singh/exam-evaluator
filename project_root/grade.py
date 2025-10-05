import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError
from ocr import file_to_part
from werkzeug.datastructures import FileStorage

# Load environment variables from .env file
load_dotenv()

# Configuration
GEMINI_MODEL = 'gemini-2.5-flash-preview-05-20'

def grade_submission(notes_file: FileStorage, answer_sheet_file: FileStorage, question_prompt: str) -> dict:
    """
    Grades a student's submission using the Gemini API based on provided notes and question.

    Args:
        notes_file: The FileStorage object for the teacher's notes.
        answer_sheet_file: The FileStorage object for the student's answer sheet.
        question_prompt: The text of the question the student answered.

    Returns:
        A dictionary containing the 'grade' and 'feedback' from the model, or an error.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"error": "GEMINI_API_KEY not found. Please check your .env file."}

    try:
        # 1. Initialize the client
        client = genai.Client(api_key=api_key)

        # 2. Convert files to multimodal Parts
        notes_part = file_to_part(notes_file)
        answer_part = file_to_part(answer_sheet_file)
        
        # 3. Define the System Instruction for precise grading
        system_instruction = (
            "You are a strict but fair academic grader. Your task is to grade a student's answer sheet "
            "based on the provided teacher's notes/study material and the original exam question. "
            "You MUST compare the keywords, concepts, and accuracy of the student's answer against the notes. "
            "The grade must be a letter (A, B, C, D, F) or a percentage (0-100). "
            "Provide detailed, constructive feedback explaining the grade, referencing where the student's answer matched "
            "or deviated from the notes. Respond ONLY with a single JSON object that strictly adheres to the provided schema."
        )

        # 4. Define the User Prompt and Content Parts
        user_prompt = (
            "Please grade the student's answer sheet. "
            f"The original exam question was: '{question_prompt}'. "
            "The first attached file is the official teacher's notes/study material. "
            "The second attached file is the student's answer. "
            "Use the notes as the definitive source for correct information."
        )
        
        content = [
            notes_part,
            answer_part,
            user_prompt
        ]
        
        # 5. Define the Structured JSON Schema for the output (MANDATORY for reliable grading)
        response_schema = {
            "type": "OBJECT",
            "properties": {
                "grade": {"type": "STRING", "description": "The final letter grade or percentage, e.g., 'A' or '92%'."},
                "feedback": {"type": "STRING", "description": "Detailed, constructive feedback based on comparison to notes."}
            },
            "required": ["grade", "feedback"]
        }

        # 6. Call the Gemini API
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=content,
            config={
                "system_instruction": system_instruction,
                "response_mime_type": "application/json",
                "response_schema": response_schema
            }
        )

        # 7. Process the JSON response
        try:
            # The model returns a JSON string in the text part
            json_text = response.text.strip()
            result = json.loads(json_text)
            return result
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response: {e}. Raw response: {response.text}")
            return {"error": f"Model returned unparseable JSON. Raw output: {response.text[:200]}..."}
        except Exception as e:
            return {"error": f"An unexpected error occurred while processing the response: {e}"}

    except APIError as e:
        return {"error": f"Gemini API Error: {e.message}"}
    except Exception as e:
        return {"error": f"An unknown error occurred: {e}"}

# Example usage (for testing purposes, not used by Flask directly)
if __name__ == '__main__':
    print("This file contains the grading logic and should be imported by app.py.")