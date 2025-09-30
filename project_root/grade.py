{
  "q1": {"answer": "Paris", "points": 1, "type": "exact"},
  "q2": {"answer": "42", "points": 1, "type": "numeric", "tolerance": 0},
  "q3": {"keywords": ["treaty", "Versailles"], "points": 2, "type": "keywords"},
  "q4": {"pattern": "^\\d{4}$", "points": 1, "type": "regex"}
}
# grading.py
import re
from difflib import SequenceMatcher

def parse_answers_from_text(text):
    # Try Q#: text lines like "q1: answer"
    pattern = re.compile(r'^\s*(?:q|question)\s*(\d+)\s*[:\)\.-]\s*(.*)$', re.I)
    answers = {}
    for line in text.splitlines():
        m = pattern.match(line)
        if m:
            qid = f"q{int(m.group(1))}"
            answers[qid] = m.group(2).strip()
    return answers

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def grade_question(student_ans, key):
    typ = key.get("type", "exact")
    points = key.get("points", 1)
    awarded = 0
    feedback = ""
    if student_ans is None:
        return 0, "no answer"
    s = student_ans.strip().lower()
    if typ == "exact":
        correct = key["answer"].strip().lower()
        if s == correct:
            awarded = points
            feedback = "exact match"
    elif typ == "numeric":
        try:
            val = float(re.sub(r'[^\d\.\-]', '', s))
            target = float(key["answer"])
            tol = float(key.get("tolerance", 0))
            if abs(val - target) <= tol:
                awarded = points
                feedback = "numeric match"
            else:
                feedback = f"numeric mismatch (got {val})"
        except Exception:
            feedback = "could not parse numeric answer"
    elif typ == "regex":
        pat = re.compile(key["pattern"], re.I)
        if pat.search(s):
            awarded = points
            feedback = "regex match"
    elif typ == "keywords":
        keywords = key.get("keywords", [])
        matches = sum(1 for kw in keywords if kw.lower() in s)
        if matches:
            awarded = round(points * matches / max(1, len(keywords)), 2)
            feedback = f"{matches}/{len(keywords)} keywords"
    elif typ == "fuzzy":
        target = key["answer"].strip().lower()
        sim = similarity(s, target)
        thresh = key.get("threshold", 0.8)
        if sim >= thresh:
            awarded = points
            feedback = f"fuzzy match (sim={sim:.2f})"
        else:
            feedback = f"low similarity (sim={sim:.2f})"
    else:
        # fallback exact
        if s == key.get("answer", "").strip().lower():
            awarded = points
            feedback = "exact fallback match"

    return awarded, feedback

def grade_response(extracted_text, answer_key):
    # extracted_text can be a string or dict of {qid: answer}
    if isinstance(extracted_text, str):
        parsed = parse_answers_from_text(extracted_text)
    elif isinstance(extracted_text, dict):
        parsed = extracted_text
    else:
        raise ValueError("extracted_text must be string or dict")

    total_points = sum(v.get("points", 1) for v in answer_key.values())
    score = 0
    details = {}

    for qid, key in answer_key.items():
        student = parsed.get(qid)
        awarded, feedback = grade_question(student, key)
        score += awarded
        details[qid] = {
            "student": student,
            "awarded": awarded,
            "max": key.get("points", 1),
            "feedback": feedback
        }

    percent = round((score / total_points) * 100, 2) if total_points else 0
    return {"score": score, "max_score": total_points, "percentage": percent, "details": details}
