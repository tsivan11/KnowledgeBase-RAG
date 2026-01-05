# Multi-format document ingestion with PDF, TXT, DOCX, HTML, CSV support.

### imports
import os
import json
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv
from loaders import load_file
### imports end

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
load_dotenv()


def get_paths(domain: str):
    if Path.cwd().name == "src":
        base = Path("..")
    else:
        base = Path(".")
    
    input_dir = base / "kb" / domain
    out_path = base / "data" / domain / "pages.jsonl"
    return input_dir, out_path


def iter_document_paths(root: Path):
    """Discover all supported document types in directory and all subdirectories."""
    supported = {".pdf", ".txt", ".md", ".docx", ".html", ".htm", ".csv", ".xlsx", ".xls", ".pptx", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}
    return sorted([f for f in root.glob("**/*") if f.is_file() and f.suffix.lower() in supported])


def ingest_documents(domain: str):
    """Ingest all documents from domain folder, write normalized pages to output."""
    input_dir, out_path = get_paths(domain)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Missing folder: {input_dir.resolve()}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = iter_document_paths(input_dir)
    if not files:
        raise FileNotFoundError(f"No documents found in: {input_dir.resolve()}")

    total_records = 0
    failed_files = []
    logger.info(f"Processing {len(files)} document(s) from {input_dir.resolve()}")

    with out_path.open("w", encoding="utf-8") as f:
        for file_path in files:
            try:
                records_extracted = 0
                for rec in load_file(file_path):
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total_records += 1
                    records_extracted += 1
                logger.info(f"  {file_path.name} ({file_path.suffix.lower()}): {records_extracted} record(s)")
            except Exception as e:
                logger.warning(f"  {file_path.name}: FAILED - {e}")
                failed_files.append(file_path.name)

    logger.info(f"Wrote {total_records} total record(s) to {out_path}")
    if failed_files:
        logger.warning(f"Failed to process: {', '.join(failed_files)}")
    else:
        logger.info("All documents processed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to ingest documents from a domain folder")
    parser.add_argument("--domain", type=str, default="contracts",
                        help="Domain folder name (default: contracts)")
    args = parser.parse_args()
    ingest_documents(args.domain)
