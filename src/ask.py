import os
import sys
import json
import logging
import argparse
from pathlib import Path

import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

# Fix encoding for Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()


def get_paths(domain: str):
    """Get index paths for a given domain."""
    if Path.cwd().name == "src":
        base = Path("..")
    else:
        base = Path(".")
    
    index_path = base / "data" / domain / "faiss.index"
    meta_path = base / "data" / domain / "chunks_meta.json"
    return index_path, meta_path

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"  # good quality, cheap
TOP_K = 8


def embed_query(client: OpenAI, query: str, max_retries: int = 3) -> np.ndarray:
    """Embed query with retry logic for API failures."""
    from time import sleep
    
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=[query])
            v = np.array(resp.data[0].embedding, dtype="float32").reshape(1, -1)
            faiss.normalize_L2(v)
            return v
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                logger.warning(f"Query embedding error (attempt {attempt + 1}/{max_retries}): {e}")
                logger.info(f"Retrying in {wait_time} seconds...")
                sleep(wait_time)
            else:
                logger.error(f"Failed to embed query after {max_retries} attempts")
                raise RuntimeError(f"Query embedding failed: {e}") from e


def main(domain: str, question: str = None):
    """Main function with comprehensive error handling."""
    try:
        index_path, meta_path = get_paths(domain)
        
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY missing. Put it in your .env file.")
        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Missing FAISS index or metadata for domain '{domain}'. Run build_index.py --domain {domain} first.")

        client = OpenAI()

        # Load index + metadata
        try:
            index = faiss.read_index(str(index_path))
            if index.ntotal == 0:
                raise ValueError(f"FAISS index is empty for domain '{domain}'")
        except Exception as e:
            raise RuntimeError(f"Failed to load FAISS index: {e}") from e
        
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if not meta:
                raise ValueError(f"Metadata is empty for domain '{domain}'")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse metadata JSON: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to load metadata: {e}") from e
        
        logger.info(f"Loaded FAISS index ({index.ntotal} vectors) and metadata ({len(meta)} records) for domain '{domain}'")

        # Read query from user or use provided question
        if question:
            query = question.strip()
        else:
            query = input("Ask a question: ").strip()
        
        if not query:
            print("No query provided.")
            return

        # Retrieve
        try:
            qvec = embed_query(client, query)
            scores, idxs = index.search(qvec, TOP_K)
            idxs = idxs[0].tolist()
            scores = scores[0].tolist()
        except Exception as e:
            raise RuntimeError(f"Failed to perform similarity search: {e}") from e

        hits = []
        for rank, (i, s) in enumerate(zip(idxs, scores), start=1):
            if i < 0 or i >= len(meta):
                logger.warning(f"Invalid index {i} returned from search")
                continue
            rec = meta[i]
            hits.append((rank, s, rec))
        
        if not hits:
            print("No relevant results found.")
            return

        # Build context blocks
        context_blocks = []
        for rank, score, rec in hits:
            source_info = f"{rec['source']} ({rec.get('source_type', 'unknown')})"
            if rec.get('page'):
                source_info += f" p.{rec['page']}"
            if rec.get('section'):
                source_info += f" s.{rec['section']}"
            context_blocks.append(
                f"[{rank}] {source_info}\n{rec['text']}"
            )
        context = "\n\n".join(context_blocks)

        system = (
            "You are a strict, citation-bound knowledgebase assistant.\n"
            "Answer ONLY using the provided context blocks. If the context is insufficient, "
            "say: 'I don't know based on the provided files.'\n"
            "Cite sources using bracket numbers like [1], [2]."
        )

        user = (
            f"CONTEXT BLOCKS:\n{context}\n\n"
            f"QUESTION: {query}\n\n"
            "Write a concise answer with citations."
        )

        try:
            resp = client.chat.completions.create(
                model=CHAT_MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except Exception as e:
            raise RuntimeError(f"Failed to generate answer: {e}") from e

        print("\n=== ANSWER ===\n")
        print(resp.choices[0].message.content)

        print("\n=== RETRIEVED (for debugging) ==\n")
        for rank, score, rec in hits:
            source_info = f"{rec['source']} ({rec.get('source_type', 'unknown')})"
            if rec.get('page'):
                source_info += f" p.{rec['page']}"
            if rec.get('section'):
                source_info += f" s.{rec['section']}"
            print(f"[{rank}] score={score:.4f} {source_info}  id={rec['chunk_id']}")
    
    except Exception as e:
        logger.error(f"Query failed for domain '{domain}': {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query a domain-specific RAG system")
    parser.add_argument("--domain", type=str, default="contracts",
                        help="Domain folder name (default: contracts)")
    parser.add_argument("--question", "-q", type=str, default=None,
                        help="Question to ask (skips interactive prompt)")
    args = parser.parse_args()
    
    main(args.domain, args.question)
