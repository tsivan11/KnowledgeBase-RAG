import json
import logging
import argparse
import re
from pathlib import Path
from typing import List

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


# ============================================================================
# CHUNKING CONFIGURATION
# ============================================================================

# Default chunking parameters
CHUNK_SIZE = 2000      # characters
CHUNK_OVERLAP = 300    # characters

# Document-type specific configurations
CHUNKING_CONFIG = {
    # Structural documents - larger chunks, respect boundaries
    "pdf": {"size": 2000, "overlap": 300, "strategy": "structural"},
    "docx": {"size": 2000, "overlap": 300, "strategy": "structural"},
    "md": {"size": 2000, "overlap": 300, "strategy": "structural"},
    "html": {"size": 2000, "overlap": 300, "strategy": "structural"},
    "pptx": {"size": 3000, "overlap": 200, "strategy": "structural"},  # slides can be longer
    
    # Unstructured text - sentence-aware splitting
    "txt": {"size": 2000, "overlap": 300, "strategy": "sentence-aware"},
    "audio": {"size": 2000, "overlap": 400, "strategy": "sentence-aware"},  # more overlap for continuity
    
    # Tabular data - smaller chunks, row-based
    "csv": {"size": 1500, "overlap": 0, "strategy": "keep-as-is"},
    "xlsx": {"size": 2500, "overlap": 0, "strategy": "keep-as-is"},
    
    # Images - already atomic
    "jpg": {"size": 10000, "overlap": 0, "strategy": "keep-as-is"},
    "jpeg": {"size": 10000, "overlap": 0, "strategy": "keep-as-is"},
    "png": {"size": 10000, "overlap": 0, "strategy": "keep-as-is"},

}

# Thresholds for keep-as-is strategy
MIN_CHUNK_SIZE = 100  # Default - don't create chunks smaller than this
MAX_ATOMIC_SIZE = 4000  # If content is smaller than this, keep as single chunk

# Document-type specific minimum sizes (override default MIN_CHUNK_SIZE)
MIN_CHUNK_SIZE_BY_TYPE = {
    "csv": 30,    # CSV rows are naturally short
    "xlsx": 30,   # Excel rows also short
}

def get_min_chunk_size(source_type: str) -> int:
    """Get minimum chunk size for a document type."""
    return MIN_CHUNK_SIZE_BY_TYPE.get(source_type, MIN_CHUNK_SIZE)


# ============================================================================
# CHUNKING STRATEGIES
# ============================================================================

def chunk_fixed_size(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Original fixed-size character-based chunking with overlap."""
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


def chunk_sentence_aware(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Sentence-aware recursive splitting.
    Tries to split on sentence boundaries, then paragraphs, then words.
    """
    text = text.strip()
    if not text:
        return []
    
    # If text is small enough, return as-is
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    
    # Define separators in order of preference
    separators = [
        "\n\n",  # Paragraph breaks
        "\n",    # Line breaks
        ". ",    # Sentence ends
        "! ",    # Exclamation sentences
        "? ",    # Question sentences
        "; ",    # Semicolons
        ", ",    # Commas
        " ",     # Words
    ]
    
    def split_text_recursive(text: str, seps: List[str]) -> List[str]:
        """Recursively split text using separators."""
        if not text.strip():
            return []
        
        # If text fits, return it
        if len(text) <= chunk_size:
            return [text]
        
        # If no more separators, fall back to character split
        if not seps:
            return chunk_fixed_size(text, chunk_size, overlap)
        
        # Try current separator
        sep = seps[0]
        parts = text.split(sep)
        
        result = []
        current_chunk = ""
        
        for i, part in enumerate(parts):
            # Re-add separator except for last part
            if i < len(parts) - 1:
                part = part + sep
            
            # If adding this part would exceed chunk_size
            if current_chunk and len(current_chunk) + len(part) > chunk_size:
                # Save current chunk
                if current_chunk.strip():
                    result.append(current_chunk.strip())
                
                # Start new chunk with overlap
                if overlap > 0 and current_chunk:
                    # Take last 'overlap' characters from previous chunk
                    overlap_text = current_chunk[-overlap:].lstrip()
                    current_chunk = overlap_text + part
                else:
                    current_chunk = part
            else:
                current_chunk += part
        
        # Add remaining chunk
        if current_chunk.strip():
            result.append(current_chunk.strip())
        
        # Recursively split chunks that are still too large
        final_result = []
        for chunk in result:
            if len(chunk) > chunk_size:
                final_result.extend(split_text_recursive(chunk, seps[1:]))
            else:
                final_result.append(chunk)
        
        return final_result
    
    return split_text_recursive(text, separators)


def chunk_structural(text: str, chunk_size: int, overlap: int, has_structure: bool) -> List[str]:
    """
    Structural chunking - respects document boundaries.
    If content already has page/section structure and is reasonable size, keep as-is.
    Otherwise, use sentence-aware splitting.
    """
    text = text.strip()
    if not text:
        return []
    
    # If content is already well-sized and has structure, keep as-is
    if has_structure and len(text) <= MAX_ATOMIC_SIZE:
        return [text]
    
    # Otherwise use sentence-aware splitting to respect natural boundaries
    return chunk_sentence_aware(text, chunk_size, overlap)


def chunk_keep_as_is(text: str, max_size: int) -> List[str]:
    """
    Keep content as single chunk if it's atomic (images, small records).
    Only split if absolutely necessary (exceeds max_size).
    """
    text = text.strip()
    if not text:
        return []
    
    # Keep as single chunk if under max size
    if len(text) <= max_size:
        return [text]
    
    # If too large, fall back to sentence-aware splitting
    logger.warning(f"Atomic content exceeds max size ({len(text)} > {max_size}), splitting...")
    return chunk_sentence_aware(text, max_size, overlap=200)


# ============================================================================
# CHUNKING DISPATCHER
# ============================================================================

def chunk_by_type(text: str, source_type: str, has_structure: bool = False) -> List[str]:
    """
    Route to appropriate chunking strategy based on document type.
    
    Args:
        text: Text to chunk
        source_type: Document type (pdf, txt, jpg, etc.)
        has_structure: Whether the record has page/section metadata
    
    Returns:
        List of text chunks
    """
    # Get configuration for this document type
    config = CHUNKING_CONFIG.get(source_type, {
        "size": CHUNK_SIZE,
        "overlap": CHUNK_OVERLAP,
        "strategy": "sentence-aware"
    })
    
    strategy = config["strategy"]
    chunk_size = config["size"]
    overlap = config["overlap"]
    
    # Route to appropriate chunker
    if strategy == "keep-as-is":
        return chunk_keep_as_is(text, chunk_size)
    elif strategy == "sentence-aware":
        return chunk_sentence_aware(text, chunk_size, overlap)
    elif strategy == "structural":
        return chunk_structural(text, chunk_size, overlap, has_structure)
    else:
        # Fallback to fixed-size
        logger.warning(f"Unknown strategy '{strategy}', using fixed-size")
        return chunk_fixed_size(text, chunk_size, overlap)


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def main(domain: str):
    in_path, out_path = get_paths(domain)
    
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input: {in_path.resolve()}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_pages = 0
    total_chunks = 0
    strategy_counts = {}  # Track which strategies are used

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            rec = json.loads(line)
            total_pages += 1

            source = rec["source"]
            source_type = rec.get("source_type", "unknown")
            page = rec.get("page")
            section = rec.get("section")
            text = rec["text"]

            # Determine if this record has structural metadata
            has_structure = page is not None or section is not None
            
            # Use document-type aware chunking
            chunks = chunk_by_type(text, source_type, has_structure)
            
            # Track strategy usage
            config = CHUNKING_CONFIG.get(source_type, {"strategy": "default"})
            strategy = config.get("strategy", "default")
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + len(chunks)
            
            for idx, chunk in enumerate(chunks):
                # Skip empty chunks or chunks below minimum size for this type
                min_size = get_min_chunk_size(source_type)
                if not chunk or len(chunk) < min_size:
                    continue
                
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
    print(f"\nChunking strategies used:")
    for strategy, count in sorted(strategy_counts.items()):
        print(f"  {strategy}: {count} chunks")
    logger.info(f"Read {total_pages} page(s) → {total_chunks} chunk(s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk pages for a domain")
    parser.add_argument("--domain", type=str, default="contracts",
                        help="Domain folder name (default: contracts)")
    args = parser.parse_args()
    
    main(args.domain)
