# ocr.py
import os
import io
from PIL import Image
import numpy as np
import cv2
import requests

OCR_PROVIDER = os.getenv("OCR_PROVIDER", "tesseract")  # "tesseract" or "remote"
OCR_API_URL = os.getenv("OCR_API_URL")
OCR_API_KEY = os.getenv("OCR_API_KEY")

def preprocess_image_bytes(image_bytes):
    # simple denoise/threshold using OpenCV
    arr = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, enc = cv2.imencode(".jpg", th)
    return enc.tobytes()

def extract_with_tesseract(image_bytes):
    try:
        from PIL import Image
        import pytesseract
    except ImportError:
        raise RuntimeError("Install pytesseract & pillow for local OCR")
    pre = preprocess_image_bytes(image_bytes)
    im = Image.open(io.BytesIO(pre))
    text = pytesseract.image_to_string(im, lang='eng')
    return text

def extract_with_remote_api(image_bytes):
    if not OCR_API_URL or not OCR_API_KEY:
        raise RuntimeError("OCR_API_URL and OCR_API_KEY must be set for remote provider")
    files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
    headers = {"Authorization": f"Bearer {OCR_API_KEY}"}
    resp = requests.post(OCR_API_URL, headers=headers, files=files, timeout=30)
    resp.raise_for_status()
    # assume API returns JSON with `text` field; adapt per provider
    data = resp.json()
    return data.get("text") or data.get("extracted_text") or resp.text

def extract_text_from_image(image_bytes: bytes) -> str:
    if OCR_PROVIDER == "tesseract":
        return extract_with_tesseract(image_bytes)
    else:
        return extract_with_remote_api(image_bytes)
