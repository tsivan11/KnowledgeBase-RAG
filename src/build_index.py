import os
import json
import time
import logging
import argparse
from pathlib import Path

import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()


def get_paths(domain: str):
    """Get input/output paths for a given domain."""
    if Path.cwd().name == "src":
        base = Path("..")
    else:
        base = Path(".")
    
    in_path = base / "data" / domain / "chunks.jsonl"
    out_index = base / "data" / domain / "faiss.index"
    out_meta = base / "data" / domain / "chunks_meta.json"
    return in_path, out_index, out_meta

EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 128  # adjust if you hit rate limits


def read_chunks(path: Path):
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def embed_texts(client: OpenAI, texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    embs = [d.embedding for d in resp.data]
    return np.array(embs, dtype="float32")


def main(domain: str):
    in_path, out_index, out_meta = get_paths(domain)
    
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY missing. Put it in your .env file.")

    if not in_path.exists():
        raise FileNotFoundError(f"Missing input: {in_path.resolve()}")

    chunks = read_chunks(in_path)
    logger.info(f"Loaded {len(chunks)} chunks from {in_path}")

    client = OpenAI()

    all_vecs = []
    meta = []

    t0 = time.time()
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        texts = [c["text"] for c in batch]

        vecs = embed_texts(client, texts)
        faiss.normalize_L2(vecs)  # cosine-ish with IndexFlatIP

        all_vecs.append(vecs)
        meta.extend([{
            "chunk_id": c["chunk_id"],
            "source": c["source"],
            "source_type": c.get("source_type", "unknown"),
            "page": c.get("page"),
            "section": c.get("section"),
            "text": c["text"],
        } for c in batch])

        print(f"Embedded {min(start + BATCH_SIZE, len(chunks))}/{len(chunks)}")

    X = np.vstack(all_vecs)
    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X)

    out_index.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_index))

    with out_meta.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    dt = time.time() - t0
    logger.info(f"Saved FAISS index to: {out_index} (vectors={index.ntotal}, dim={dim})")
    logger.info(f"Saved metadata to:   {out_meta} (records={len(meta)})")
    logger.info(f"Done in {dt:.1f}s")
    print(f"\nSaved FAISS index to: {out_index} (vectors={index.ntotal}, dim={dim})")
    print(f"Saved metadata to:   {out_meta} (records={len(meta)})")
    print(f"Done in {dt:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index for a domain")
    parser.add_argument("--domain", type=str, default="contracts",
                        help="Domain folder name (default: contracts)")
    args = parser.parse_args()
    
    main(args.domain)
