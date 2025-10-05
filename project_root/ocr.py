import os
import io
from google.genai.types import Part
from werkzeug.datastructures import FileStorage

def get_mime_type(filename):
    """
    Determines the MIME type based on the file extension.
    This is necessary for the Gemini API to correctly interpret the file data.
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext in ['.jpg', '.jpeg']:
        return 'image/jpeg'
    elif ext == '.png':
        return 'image/png'
    elif ext == '.pdf':
        # The API supports PDF directly as application/pdf for multimodal analysis.
        return 'application/pdf'
    else:
        # Default to octet-stream for unknown types
        return 'application/octet-stream'

def file_to_part(uploaded_file: FileStorage) -> Part:
    """
    Converts an uploaded Werkzeug FileStorage object into a Gemini API Part object,
    reading the file content into memory and assigning the correct MIME type.

    Args:
        uploaded_file: The file object from Flask's request.files.

    Returns:
        A google.genai.types.Part object for the API request.
    """
    # Read file content into memory
    # We must read the file here since it's an in-memory object stream.
    file_bytes = uploaded_file.read()

    # Determine MIME type
    mime_type = get_mime_type(uploaded_file.filename)

    # Reset file pointer to the beginning for safety, in case Flask needs it later
    uploaded_file.seek(0)
    
    return Part.from_bytes(
        data=file_bytes,
        mime_type=mime_type
    )
