import os
import json
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from grade import grade_submission

# --- Initialization and Configuration ---
app = Flask(__name__, static_url_path='/static')
# In a real application, you would set a secret key: app.secret_key = os.environ.get("SECRET_KEY")

# For file handling, we'll use a temporary approach of reading the file directly from the request
# instead of saving to a file system folder, as the 'uploads' folder might not be writable in all environments.
# Since the files are small and passed directly to the API, this is safer.
# We will use the in-memory FileStorage objects directly.

# --- Routes ---

# app.py (Corrected)

@app.route('/')
def home():
    """Renders the main grading input form."""
    # Your actual code to render a template goes here
    # e.g., return render_template('index.html')
    return render_template('grading_input.html')

@app.route('/grade', methods=['POST'])
def grade():
    """Handles the form submission, calls the grading logic, and renders the result."""
    
    # 1. Get Inputs
    notes_file = request.files.get('notes_file')
    answer_sheet_file = request.files.get('answer_sheet_file')
    question_prompt = request.form.get('question_prompt', '').strip()

    # Simple validation (client-side validation should also exist)
    if not notes_file or not answer_sheet_file or not question_prompt:
        return render_template('grading_input.html', 
                               error="All three inputs (Notes file, Answer file, and Question text) are required.")

    # 2. Call the Gemini Grading Service
    # We pass the in-memory FileStorage objects directly
    try:
        grade_result = grade_submission(notes_file, answer_sheet_file, question_prompt)
    except Exception as e:
        print(f"FATAL ERROR during grading: {e}")
        grade_result = {"error": "A server error occurred during the AI grading process."}


    # 3. Render the Result
    if "error" in grade_result:
        # Render the page with the error message
        return render_template('grading_input.html', error=grade_result["error"])
    else:
        # Render the page with the successful grade and feedback
        return render_template('grading_input.html', grade_result=grade_result)

# --- Entry Point ---
if __name__ == '__main__':
    # In a production environment, you would use a WSGI server (like Gunicorn)
    # app.run() is fine for local development
    # Ensure a 'static' folder exists for style.css, or use the root for style.css if deployed differently
    
    # Check if style.css is accessible in the 'static' folder for Flask
    # Create a simple placeholder for 'static/style.css' to match the template URL if needed,
    # though style.css is in the root in the provided structure.
    # Flask typically expects static files in a 'static' folder.
    # We will adjust to place style.css in the root for simplicity in this setup.
    # We will need to make sure the style.css is accessible, which it is, as its in the root.
    
    # Start the application
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
 