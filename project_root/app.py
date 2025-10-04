# app.py
import os
import json
from flask import Flask, request, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# --- Gemini Imports ---
import google.generativeai as genai
from PIL import Image
import io

# Load environment variables (must be called before accessing the key)
load_dotenv() 

# --- Configure Gemini Client ---
# The client will automatically pick up the key from the environment variable
try:
    client = genai.Client()
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    # Handle the case where the key is missing or invalid
    # For now, we'll let Flask run and handle the error at the endpoint.

app = Flask(__name__)

# Define upload folder and file types (Crucial for security and file handling)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok"), 200

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify(error="file is required"), 400
    f = request.files["file"]
    image_bytes = f.read()
    try:
        text = extract_text_from_image(image_bytes)
    except Exception as e:
        return jsonify(error=str(e)), 500
    return jsonify(extracted_text=text), 200

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/evaluate', methods=['POST'])
def evaluate():
    if 'answer_sheet' not in request.files:
        return jsonify({"error": "No answer sheet file part"}), 400
    
    answer_sheet = request.files['answer_sheet']
    answer_key_text = request.form.get('answer_key', '')

    if answer_sheet.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if answer_key_text == '':
        return jsonify({"error": "Answer key is missing"}), 400

    if answer_sheet and allowed_file(answer_sheet.filename):
        # We don't need to save the file to disk, we can pass the stream directly
        # to PIL and then to Gemini, which is more secure and efficient.
        file_stream = io.BytesIO(answer_sheet.read())
        
        # Call the new grading function
        grade_result, status_code = grade_with_gemini(file_stream, answer_key_text)
        
        # Return the structured JSON result to the frontend
        return jsonify(grade_result), status_code
    
    return jsonify({"error": "File type not allowed"}), 400

@app.route("/", methods=["GET", "POST"])
def grading_input():
    result = None
    if request.method == "POST":
        file = request.files.get("file")
        answer_key_raw = request.form.get("answer_key")
        use_gemini = request.form.get("use_gemini")  # Checkbox in your form
        if not file or not answer_key_raw:
            result = "Both file and answer key are required."
        else:
            import json
            try:
                answer_key = json.loads(answer_key_raw)
                image_bytes = file.read()
                if use_gemini:
                    graded, status = grade_with_gemini(io.BytesIO(image_bytes), answer_key)
                    result = graded if status == 200 else graded.get("error", "Gemini grading failed.")
                else:
                    extracted_text = extract_text_from_image(image_bytes)
                    graded = grade_response(extracted_text, answer_key)
                    import pprint
                    result = pprint.pformat(graded)
            except Exception as e:
                result = f"Error: {str(e)}"
    return render_template("grading_input.html", result=result)

def grade_with_gemini(file_stream, answer_key):
    """
    Sends the image and answer key to Gemini for grading and returns a JSON result.
    """
    if not os.getenv("GEMINI_API_KEY"):
        return {"error": "API Key not configured."}, 500

    try:
        # Load the image/PDF from the file stream using PIL
        uploaded_file = Image.open(file_stream)
        
        # --- 1. Define the Structured JSON Schema for predictable output ---
        grading_schema = genai.Schema(
            type=genai.Type.OBJECT,
            properties={
                "total_score": genai.Schema(type=genai.Type.NUMBER, description="The final numerical score."),
                "max_score": genai.Schema(type=genai.Type.NUMBER, description="The total possible score."),
                "results": genai.Schema(
                    type=genai.Type.ARRAY,
                    description="An array of results for each question.",
                    items=genai.Schema(
                        type=genai.Type.OBJECT,
                        properties={
                            "question_number": genai.Schema(type=genai.Type.STRING),
                            "correct_answer": genai.Schema(type=genai.Type.STRING),
                            "student_answer": genai.Schema(type=genai.Type.STRING, description="The answer extracted from the image."),
                            "is_correct": genai.Schema(type=genai.Type.BOOLEAN),
                            "feedback": genai.Schema(type=genai.Type.STRING)
                        }
                    )
                )
            }
        )
        
        # --- 2. Craft the Multimodal Prompt ---
        prompt = f"""
        You are an expert, meticulous exam grader. Your task is to extract, compare, and grade a student's answer sheet provided in the image.

        **Answer Key:**
        {answer_key}

        **Grading Rules:**
        1. Analyze the attached image to find the student's response for each question.
        2. Compare the extracted student's response to the provided Answer Key.
        3. Assign a score of 1 for each correct answer and 0 for each incorrect answer. The total score is the sum of points.
        4. Provide brief, actionable feedback for each question in the 'feedback' field.
        5. **CRITICAL:** You MUST return the output as a valid JSON object that strictly conforms to the provided schema. Do not include any other text, markdown, or commentary outside of the JSON.
        """

        # --- 3. Call the Gemini API ---
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=[prompt, uploaded_file],
            config=genai.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=grading_schema,
                temperature=0.0 # Crucial for factual, non-creative grading
            )
        )
        
        # The response.text is the JSON string
        return json.loads(response.text), 200

    except Exception as e:
        # Catch any API or processing errors
        return {"error": f"An error occurred during grading: {str(e)}"}, 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5050)
