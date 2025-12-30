from PIL import Image
import pytesseract
from pathlib import Path

# 🔑 Explicit path (fixes Git Bash / venv issues)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

BASE_DIR = Path(__file__).parent
img = Image.open(BASE_DIR / "test.jpg")

text = pytesseract.image_to_string(img, config="--psm 6")
print(text[:500])
