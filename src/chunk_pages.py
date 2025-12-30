import json
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_paths(domain: str):
    """Get input/output paths for a given domain."""
    if Path.cwd().name == "src":
        base = Path("..")
    else:
        base = Path(".")
    
    in_path = base / "data" / domain / "pages.jsonl"
    out_path = base / "data" / domain / "chunks.jsonl"
    return in_path, out_path

# Contract-friendly defaults
CHUNK_SIZE = 2000      # characters
CHUNK_OVERLAP = 300    # characters


def chunk_text(text: str, chunk_size: int, overlap: int):
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def main(domain: str):
    in_path, out_path = get_paths(domain)
    
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input: {in_path.resolve()}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_pages = 0
    total_chunks = 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            total_pages += 1

            source = rec["source"]
            source_type = rec.get("source_type", "unknown")
            page = rec.get("page")
            section = rec.get("section")
            text = rec["text"]

            chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
            for idx, chunk in enumerate(chunks):
                # Build chunk_id with available location info
                location_parts = [source]
                if page is not None:
                    location_parts.append(f"p{page}")
                if section is not None:
                    location_parts.append(f"s{section}")
                location_parts.append(f"c{idx}")
                chunk_id = "::".join(location_parts)
                
                out = {
                    "chunk_id": chunk_id,
                    "source": source,
                    "source_type": source_type,
                    "page": page,
                    "section": section,
                    "text": chunk,
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"Read {total_pages} page(s) from {in_path}")
    print(f"Wrote {total_chunks} chunk(s) to {out_path}")
    print(f"Chunk params: size={CHUNK_SIZE} chars, overlap={CHUNK_OVERLAP} chars")
    logger.info(f"Read {total_pages} page(s) → {total_chunks} chunk(s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk pages for a domain")
    parser.add_argument("--domain", type=str, default="contracts",
                        help="Domain folder name (default: contracts)")
    args = parser.parse_args()
    
    main(args.domain)
