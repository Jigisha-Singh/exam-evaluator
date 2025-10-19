# ocr.py

import os
import io
from google import genai
from google.genai.types import Part
from werkzeug.datastructures import FileStorage
from typing import Dict

# Initialize the Gemini Client globally. It will automatically use the 
# GEMINI_API_KEY environment variable.
try:
    client = genai.Client()
except Exception as e:
    print(f"ERROR: Failed to initialize Gemini client. Is GEMINI_API_KEY set? {e}")
    client = None

def get_mime_type(filename):
    """
    Determines the MIME type based on the file extension.
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext in ['.jpg', '.jpeg']:
        return 'image/jpeg'
    elif ext == '.png':
        return 'image/png'
    elif ext == '.pdf':
        return 'application/pdf'
    else:
        return 'application/octet-stream'

def file_to_part(uploaded_file: FileStorage) -> Part:
    """
    Converts a FileStorage object into a Gemini API Part object.
    """
    # Read file content into memory
    file_bytes = uploaded_file.read()

    # Determine MIME type
    mime_type = get_mime_type(uploaded_file.filename)

    # Reset file pointer for safety (optional in this context but good practice)
    uploaded_file.seek(0)
    
    return Part.from_bytes(
        data=file_bytes,
        mime_type=mime_type
    )

def run_ocr(uploaded_file: FileStorage) -> Dict[str, str]:
    """
    Performs OCR and extraction on the uploaded exam paper using the Gemini API.

    The function now returns a DICT to easily integrate with your grading logic.
    {
        "Q1_text": "...",
        "Q2_text": "...",
        ...
    }

    Args:
        uploaded_file: The file object from Flask's request.files.

    Returns:
        A dictionary where keys are question IDs and values are extracted strings,
        or an error dictionary if the API call fails.
    """
    if client is None:
        # Fallback for testing without API key - return mock text
        return f"Mock extracted text from {uploaded_file.filename}: This is a sample student answer for testing purposes."

    # 1. Convert the file object for the API
    file_part = file_to_part(uploaded_file)
    
    # 2. Define the prompt for structured extraction
    # This prompt tells Gemini exactly what to do and how to format the output.
    system_prompt = (
        "You are an expert OCR and text extraction engine for educational exams. "
        "Format your entire output as a single, contiguous string. "
    )
    
    

    # 3. Call the Gemini API
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash', # Fast and effective for OCR/Extraction tasks
            contents=[ file_part],
            config={"system_instruction": system_prompt}
        )
        
        # 4. Process the raw string output into a clean dictionary
        # The model output will be a single string like: "Q1_text: The Earth is flat\nQ2_text: I don't know"
        # Access the response text correctly for the current API version
        if hasattr(response, 'text'):
            raw_text = response.text
        elif hasattr(response, 'candidates') and response.candidates:
            raw_text = response.candidates[0].content.parts[0].text
        else:
            raw_text = str(response)

        return raw_text

    except Exception as e:
        return {"error": f"Gemini API call failed: {e}"}