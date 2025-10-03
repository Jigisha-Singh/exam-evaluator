# app.py
import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from ocr import extract_text_from_image
from grade import grade_response

app = Flask(__name__)

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

@app.route("/evaluate", methods=["POST"])
def evaluate():
    # Accept JSON body or multipart (file + answer_key JSON)
    if request.is_json:
        data = request.get_json()
        extracted_text = data.get("extracted_text")
        answer_key = data.get("answer_key")
    else:
        # multipart form-data: optional file, and answer_key as JSON string
        answer_key_raw = request.form.get("answer_key")
        answer_key = None
        if answer_key_raw:
            import json
            answer_key = json.loads(answer_key_raw)
        if "file" in request.files:
            extracted_text = extract_text_from_image(request.files["file"].read())
        else:
            extracted_text = request.form.get("extracted_text")

    if not answer_key:
        return jsonify(error="answer_key required"), 400
    if not extracted_text:
        return jsonify(error="extracted_text or file required"), 400

    result = grade_response(extracted_text, answer_key)
    return jsonify(result), 200

@app.route("/", methods=["GET", "POST"])
def grading_input():
    result = None
    if request.method == "POST":
        file = request.files.get("file")
        answer_key_raw = request.form.get("answer_key")
        if not file or not answer_key_raw:
            result = "Both file and answer key are required."
        else:
            import json
            try:
                answer_key = json.loads(answer_key_raw)
                extracted_text = extract_text_from_image(file.read())
                graded = grade_response(extracted_text, answer_key)
                import pprint
                result = pprint.pformat(graded)
            except Exception as e:
                result = f"Error: {str(e)}"
    return render_template("grading_input.html", result=result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
