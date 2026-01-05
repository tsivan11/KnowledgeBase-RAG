# src/loaders.py
"""
Multi-format document loaders.
Each loader yields normalized records: {source, source_type, section, page, text}
"""
import json
import logging
from pathlib import Path
from typing import Iterator, Dict, Any

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import fitz
except ImportError:
    fitz = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract = None
    Image = None

try:
    from docx import Document
except ImportError:
    Document = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from pptx import Presentation
except ImportError:
    Presentation = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None


def _ocr_page(page: "fitz.Page") -> str:
    """OCR a PDF page if pytesseract is available."""
    if pytesseract is None or Image is None:
        return ""
    
    import os
    tess_cmd = os.getenv("TESSERACT_CMD")
    if tess_cmd:
        pytesseract.pytesseract.tesseract_cmd = tess_cmd
    else:
        default_win = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
        if os.path.exists(default_win):
            pytesseract.pytesseract.tesseract_cmd = default_win
    
    try:
        pix = page.get_pixmap(dpi=200, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text = pytesseract.image_to_string(img)
        return (text or "").strip()
    except Exception as e:
        logger.debug(f"OCR failed: {e}")
        return ""


def load_pdf(file_path: Path) -> Iterator[Dict[str, Any]]:
    """Load PDF with table extraction, OCR fallback for scanned pages."""
    if fitz is None:
        logger.warning(f"PyMuPDF not installed, skipping {file_path}")
        return
    
    source = file_path.name
    
    # Try table extraction with pdfplumber first
    if pdfplumber is not None:
        try:
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    page_text_parts = []
                    
                    # Extract tables
                    tables = page.extract_tables()
                    if tables:
                        for table_idx, table in enumerate(tables):
                            formatted_table = _format_table(table)
                            if formatted_table:
                                page_text_parts.append(f"[Table {table_idx + 1}]\n{formatted_table}")
                    
                    # Extract remaining text (non-table content)
                    regular_text = page.extract_text()
                    if regular_text:
                        page_text_parts.append(regular_text.strip())
                    
                    # Combine tables and text
                    combined_text = "\n\n".join(page_text_parts).strip()
                    
                    if combined_text:
                        yield {
                            "source": source,
                            "source_type": "pdf",
                            "page": i,
                            "section": None,
                            "text": combined_text,
                        }
            return  # Successfully processed with pdfplumber
        except Exception as e:
            logger.debug(f"pdfplumber extraction failed, falling back to PyMuPDF: {e}")
    
    # Fallback to PyMuPDF for regular text extraction
    try:
        doc = fitz.open(file_path)
        
        for i in range(len(doc)):
            page = doc.load_page(i)
            text = (page.get_text("text") or "").strip()
            
            if not text:
                text = _ocr_page(page)
            
            if not text:
                continue
            
            yield {
                "source": source,
                "source_type": "pdf",
                "page": i + 1,
                "section": None,
                "text": text,
            }
        
        doc.close()
    except Exception as e:
        logger.warning(f"Failed to load PDF {file_path}: {e}")


def load_txt(file_path: Path) -> Iterator[Dict[str, Any]]:
    """Load plain text file; treat each paragraph as a section."""
    try:
        content = file_path.read_text(encoding="utf-8")
        source = file_path.name
        
        # Split by double newlines (paragraphs)
        sections = [p.strip() for p in content.split("\n\n") if p.strip()]
        
        for idx, text in enumerate(sections, start=1):
            yield {
                "source": source,
                "source_type": "txt",
                "page": None,
                "section": idx,
                "text": text,
            }
    except Exception as e:
        logger.warning(f"Failed to load TXT {file_path}: {e}")


def load_md(file_path: Path) -> Iterator[Dict[str, Any]]:
    """Load Markdown file; preserve headings as section metadata."""
    try:
        content = file_path.read_text(encoding="utf-8")
        source = file_path.name
        
        # Simple approach: split by headings (# ## ###)
        import re
        parts = re.split(r"^(#{1,6}\s+.+)$", content, flags=re.MULTILINE)
        
        section_idx = 1
        current_heading = None
        
        for part in parts:
            if not part.strip():
                continue
            
            # Check if it's a heading
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", part)
            if heading_match:
                current_heading = heading_match.group(2).strip()
            else:
                text = part.strip()
                if text:
                    yield {
                        "source": source,
                        "source_type": "md",
                        "page": None,
                        "section": current_heading or f"section_{section_idx}",
                        "text": text,
                    }
                    section_idx += 1
    except Exception as e:
        logger.warning(f"Failed to load MD {file_path}: {e}")


def load_docx(file_path: Path) -> Iterator[Dict[str, Any]]:
    """Load DOCX file; extract paragraphs."""
    if Document is None:
        logger.warning(f"python-docx not installed, skipping {file_path}")
        return
    
    try:
        doc = Document(file_path)
        source = file_path.name
        
        for idx, para in enumerate(doc.paragraphs, start=1):
            text = para.text.strip()
            if text:
                yield {
                    "source": source,
                    "source_type": "docx",
                    "page": None,
                    "section": idx,
                    "text": text,
                }
    except Exception as e:
        logger.warning(f"Failed to load DOCX {file_path}: {e}")


def load_html(file_path: Path) -> Iterator[Dict[str, Any]]:
    """Load HTML file; extract text and preserve heading hierarchy."""
    if BeautifulSoup is None:
        logger.warning(f"BeautifulSoup4 not installed, skipping {file_path}")
        return
    
    try:
        content = file_path.read_text(encoding="utf-8")
        soup = BeautifulSoup(content, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        source = file_path.name
        section_idx = 1
        current_heading = None
        
        # Walk through body or full soup
        for elem in (soup.body or soup).descendants:
            if isinstance(elem, str):
                text = elem.strip()
                if text and len(text) > 1:  # Avoid single chars
                    yield {
                        "source": source,
                        "source_type": "html",
                        "page": None,
                        "section": current_heading or f"section_{section_idx}",
                        "text": text,
                    }
                    section_idx += 1
            elif hasattr(elem, "name"):
                if elem.name in ["h1", "h2", "h3"]:
                    current_heading = elem.get_text().strip()
    except Exception as e:
        logger.warning(f"Failed to load HTML {file_path}: {e}")


def load_csv(file_path: Path) -> Iterator[Dict[str, Any]]:
    """Load CSV file; flatten rows with column context."""
    if pd is None:
        logger.warning(f"pandas not installed, skipping {file_path}")
        return
    
    try:
        df = pd.read_csv(file_path)
        source = file_path.name
        
        for idx, row in df.iterrows():
            # Format row as key=value pairs
            text_parts = []
            for col, val in row.items():
                if pd.notna(val):
                    text_parts.append(f"{col}: {val}")
            
            text = " | ".join(text_parts)
            if text.strip():
                yield {
                    "source": source,
                    "source_type": "csv",
                    "page": None,
                    "section": f"row_{idx + 2}",  # +2 for header and 0-indexing
                    "text": text,
                }
    except Exception as e:
        logger.warning(f"Failed to load CSV {file_path}: {e}")


def load_xlsx(file_path: Path) -> Iterator[Dict[str, Any]]:
    """Load Excel file; process each sheet and row with column context."""
    if pd is None:
        logger.warning(f"pandas not installed, skipping {file_path}")
        return
    
    try:
        # Read all sheets
        excel_file = pd.ExcelFile(file_path)
        source = file_path.name
        
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            
            for idx, row in df.iterrows():
                # Format row as key=value pairs
                text_parts = []
                for col, val in row.items():
                    if pd.notna(val):
                        text_parts.append(f"{col}: {val}")
                
                text = " | ".join(text_parts)
                if text.strip():
                    yield {
                        "source": source,
                        "source_type": "xlsx",
                        "page": None,
                        "section": f"{sheet_name}::row_{idx + 2}",  # +2 for header and 0-indexing
                        "text": text,
                    }
    except Exception as e:
        logger.warning(f"Failed to load Excel {file_path}: {e}")


def load_pptx(file_path: Path) -> Iterator[Dict[str, Any]]:
    """Load PowerPoint file; extract text from slides."""
    if Presentation is None:
        logger.warning(f"python-pptx not installed, skipping {file_path}")
        return
    
    try:
        prs = Presentation(file_path)
        source = file_path.name
        
        for slide_idx, slide in enumerate(prs.slides, start=1):
            text_parts = []
            
            # Extract text from all shapes
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    text_parts.append(shape.text.strip())
            
            text = "\n".join(text_parts)
            if text.strip():
                yield {
                    "source": source,
                    "source_type": "pptx",
                    "page": slide_idx,
                    "section": None,
                    "text": text,
                }
    except Exception as e:
        logger.warning(f"Failed to load PowerPoint {file_path}: {e}")


def load_image(file_path: Path) -> Iterator[Dict[str, Any]]:
    """Load image file and extract text via OCR."""
    if pytesseract is None or Image is None:
        logger.warning(f"pytesseract or Pillow not installed, skipping {file_path}")
        return
    
    import os
    tess_cmd = os.getenv("TESSERACT_CMD")
    if tess_cmd:
        pytesseract.pytesseract.tesseract_cmd = tess_cmd
    else:
        default_win = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(default_win):
            pytesseract.pytesseract.tesseract_cmd = default_win
    
    try:
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
        text = (text or "").strip()
        
        if text:
            yield {
                "source": file_path.name,
                "source_type": file_path.suffix.lower()[1:],  # e.g., "jpg", "png"
                "page": None,
                "section": None,
                "text": text,
            }
    except Exception as e:
        logger.warning(f"Failed to load image {file_path}: {e}")


def load_audio(file_path: Path) -> Iterator[Dict[str, Any]]:
    """Transcribe audio file using OpenAI Whisper API."""
    if OpenAI is None:
        logger.warning(f"OpenAI not installed, skipping {file_path}")
        return
    
    import os
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning(f"OPENAI_API_KEY not set, skipping {file_path}")
        return
    
    try:
        client = OpenAI()
        
        # Check file size (Whisper API limit is 25MB)
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 25:
            logger.warning(f"Audio file {file_path.name} is {file_size_mb:.1f}MB, exceeds 25MB limit")
            return
        
        logger.info(f"Transcribing audio file: {file_path.name} ({file_size_mb:.1f}MB)...")
        
        with open(file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        
        # Return full transcript as single page
        # Let the chunker handle splitting intelligently
        if transcript and transcript.strip():
            yield {
                "source": file_path.name,
                "source_type": "audio",
                "page": None,
                "section": None,
                "text": transcript.strip(),
            }
        
        logger.info(f"Successfully transcribed {file_path.name}")
        
    except Exception as e:
        logger.warning(f"Failed to transcribe audio {file_path}: {e}")


# Loader registry
LOADERS = {
    ".pdf": load_pdf,
    ".txt": load_txt,
    ".md": load_md,
    ".docx": load_docx,
    ".html": load_html,
    ".htm": load_html,
    ".csv": load_csv,
    ".xlsx": load_xlsx,
    ".xls": load_xlsx,
    ".pptx": load_pptx,
    ".jpg": load_image,
    ".jpeg": load_image,
    ".png": load_image,
    ".tiff": load_image,
    ".tif": load_image,
    ".bmp": load_image,
    # Audio formats
    ".mp3": load_audio,
    ".mp4": load_audio,
    ".mpeg": load_audio,
    ".mpga": load_audio,
    ".m4a": load_audio,
    ".wav": load_audio,
    ".webm": load_audio,
}


def load_file(file_path: Path) -> Iterator[Dict[str, Any]]:
    """Auto-detect format and load file."""
    ext = file_path.suffix.lower()
    
    if ext not in LOADERS:
        logger.warning(f"Unsupported file type: {ext} ({file_path.name})")
        return
    
    loader = LOADERS[ext]
    yield from loader(file_path)
