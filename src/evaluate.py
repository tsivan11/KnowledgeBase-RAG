"""
RAG Evaluation Script

Evaluates RAG system quality using:
- Faithfulness: Does the answer stick to retrieved context?
- Answer Relevance: Does it answer the question?
- Context Precision: Are retrieved chunks relevant?
- Citation Quality: Are sources properly cited?

Usage:
    python src/evaluate.py --domain astronomy --test-file eval/test_questions.json
    python src/evaluate.py --domain astronomy --interactive
"""
import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

# Models
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o-mini"  # Can upgrade to gpt-4 for better evaluation
TOP_K = 8


def get_paths(domain: str):
    """Get index paths for a given domain."""
    if Path.cwd().name == "src":
        base = Path("..")
    else:
        base = Path(".")
    
    index_path = base / "data" / domain / "faiss.index"
    meta_path = base / "data" / domain / "chunks_meta.json"
    return index_path, meta_path


def query_rag(client: OpenAI, domain: str, question: str) -> Dict[str, Any]:
    """Query the RAG system and return answer with retrieved context."""
    index_path, meta_path = get_paths(domain)
    
    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Domain '{domain}' not indexed")
    
    # Load index and metadata
    index = faiss.read_index(str(index_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    
    # Embed query
    resp = client.embeddings.create(model=EMBED_MODEL, input=[question])
    qvec = np.array(resp.data[0].embedding, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(qvec)
    
    # Search
    scores, idxs = index.search(qvec, TOP_K)
    idxs = idxs[0].tolist()
    scores = scores[0].tolist()
    
    # Get retrieved chunks
    retrieved_contexts = []
    for rank, (i, s) in enumerate(zip(idxs, scores), start=1):
        if i < 0 or i >= len(meta):
            continue
        rec = meta[i]
        retrieved_contexts.append({
            "rank": rank,
            "score": float(s),
            "text": rec["text"],
            "source": rec["source"],
            "page": rec.get("page"),
        })
    
    # Build context for generation
    context_blocks = []
    for ctx in retrieved_contexts:
        source_info = f"{ctx['source']}"
        if ctx.get('page'):
            source_info += f" p.{ctx['page']}"
        context_blocks.append(f"[{ctx['rank']}] {source_info}\n{ctx['text']}")
    
    context = "\n\n".join(context_blocks)
    
    # Generate answer
    system = (
        "You are a strict, citation-bound assistant. "
        "Answer ONLY using the provided context blocks. If the context is insufficient, "
        "say: 'I don't know based on the provided documents.' "
        "Cite sources using bracket numbers like [1], [2]."
    )
    
    user = (
        f"CONTEXT BLOCKS:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "Write a concise answer with citations."
    )
    
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    
    answer = resp.choices[0].message.content
    
    return {
        "question": question,
        "answer": answer,
        "contexts": retrieved_contexts,
        "context_text": context,
    }


def evaluate_faithfulness(client: OpenAI, question: str, answer: str, contexts: List[Dict]) -> Dict[str, Any]:
    """Evaluate if the answer is faithful to the retrieved context."""
    context_text = "\n\n".join([f"[{c['rank']}] {c['text']}" for c in contexts])
    
    prompt = f"""Evaluate if the ANSWER is faithful to the CONTEXT (i.e., doesn't hallucinate or add information not in the context).

CONTEXT:
{context_text}

QUESTION:
{question}

ANSWER:
{answer}

Rate the faithfulness on a scale of 1-5:
1 - Completely unfaithful (hallucinated, contradicts context)
2 - Mostly unfaithful (significant unsupported claims)
3 - Partially faithful (some unsupported details)
4 - Mostly faithful (minor additions)
5 - Completely faithful (all claims supported by context)

Respond in JSON format:
{{
    "score": <1-5>,
    "reasoning": "<brief explanation>",
    "unsupported_claims": ["<claim1>", "<claim2>", ...]
}}"""
    
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    result = json.loads(resp.choices[0].message.content)
    return result


def evaluate_answer_relevance(client: OpenAI, question: str, answer: str) -> Dict[str, Any]:
    """Evaluate if the answer actually addresses the question."""
    prompt = f"""Evaluate if the ANSWER actually addresses the QUESTION.

QUESTION:
{question}

ANSWER:
{answer}

Rate the relevance on a scale of 1-5:
1 - Completely irrelevant (doesn't address question at all)
2 - Mostly irrelevant (tangentially related)
3 - Partially relevant (addresses some aspects)
4 - Mostly relevant (addresses main points)
5 - Completely relevant (directly and fully answers question)

Respond in JSON format:
{{
    "score": <1-5>,
    "reasoning": "<brief explanation>"
}}"""
    
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    result = json.loads(resp.choices[0].message.content)
    return result


def evaluate_context_precision(client: OpenAI, question: str, contexts: List[Dict]) -> Dict[str, Any]:
    """Evaluate if the retrieved contexts are relevant to the question."""
    context_text = "\n\n".join([f"[{c['rank']}] {c['text']}" for c in contexts])
    
    prompt = f"""Evaluate if the RETRIEVED CONTEXTS are relevant to answering the QUESTION.

QUESTION:
{question}

RETRIEVED CONTEXTS:
{context_text}

Rate the context precision on a scale of 1-5:
1 - None of the contexts are relevant
2 - Few contexts are relevant
3 - About half are relevant
4 - Most contexts are relevant
5 - All or nearly all contexts are relevant

Respond in JSON format:
{{
    "score": <1-5>,
    "reasoning": "<brief explanation>",
    "relevant_count": <number of relevant contexts out of {len(contexts)}>
}}"""
    
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    result = json.loads(resp.choices[0].message.content)
    return result


def evaluate_citation_quality(answer: str, contexts: List[Dict]) -> Dict[str, Any]:
    """Evaluate if sources are properly cited."""
    import re
    
    # Find all citations like [1], [2], etc.
    citations = re.findall(r'\[(\d+)\]', answer)
    unique_citations = set(citations)
    
    # Check if citations are valid (within range of retrieved contexts)
    valid_citations = [int(c) for c in unique_citations if 1 <= int(c) <= len(contexts)]
    invalid_citations = [int(c) for c in unique_citations if int(c) > len(contexts)]
    
    # Check if answer says "I don't know"
    dont_know = any(phrase in answer.lower() for phrase in [
        "i don't know",
        "i do not know",
        "cannot answer",
        "insufficient",
        "not provided"
    ])
    
    if dont_know:
        score = 5 if len(citations) == 0 else 3  # Should not cite if saying "I don't know"
        reasoning = "Appropriately stated lack of knowledge" if len(citations) == 0 else "Said 'I don't know' but still cited sources"
    else:
        if len(valid_citations) == 0:
            score = 1
            reasoning = "No citations provided"
        elif len(invalid_citations) > 0:
            score = 2
            reasoning = f"Some invalid citations: {invalid_citations}"
        elif len(valid_citations) >= 2:
            score = 5
            reasoning = f"Good citation coverage: {len(valid_citations)} sources cited"
        else:
            score = 4
            reasoning = "At least one citation provided"
    
    return {
        "score": score,
        "reasoning": reasoning,
        "citation_count": len(valid_citations),
        "total_citations": len(citations),
        "invalid_citations": invalid_citations,
    }


def evaluate_single_question(client: OpenAI, domain: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate a single test case."""
    question = test_case["question"]
    expected_answer = test_case.get("expected_answer")  # Optional
    
    logger.info(f"Evaluating: {question}")
    
    # Query RAG
    try:
        rag_result = query_rag(client, domain, question)
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        return {
            "question": question,
            "error": str(e),
            "scores": {}
        }
    
    # Evaluate
    scores = {}
    
    # 1. Faithfulness
    try:
        faithfulness = evaluate_faithfulness(
            client, question, rag_result["answer"], rag_result["contexts"]
        )
        scores["faithfulness"] = faithfulness
    except Exception as e:
        logger.warning(f"Faithfulness evaluation failed: {e}")
        scores["faithfulness"] = {"error": str(e)}
    
    # 2. Answer Relevance
    try:
        relevance = evaluate_answer_relevance(
            client, question, rag_result["answer"]
        )
        scores["answer_relevance"] = relevance
    except Exception as e:
        logger.warning(f"Answer relevance evaluation failed: {e}")
        scores["answer_relevance"] = {"error": str(e)}
    
    # 3. Context Precision
    try:
        precision = evaluate_context_precision(
            client, question, rag_result["contexts"]
        )
        scores["context_precision"] = precision
    except Exception as e:
        logger.warning(f"Context precision evaluation failed: {e}")
        scores["context_precision"] = {"error": str(e)}
    
    # 4. Citation Quality
    try:
        citations = evaluate_citation_quality(
            rag_result["answer"], rag_result["contexts"]
        )
        scores["citation_quality"] = citations
    except Exception as e:
        logger.warning(f"Citation evaluation failed: {e}")
        scores["citation_quality"] = {"error": str(e)}
    
    return {
        "question": question,
        "answer": rag_result["answer"],
        "expected_answer": expected_answer,
        "retrieved_contexts": len(rag_result["contexts"]),
        "scores": scores,
    }


def print_results(results: List[Dict[str, Any]]):
    """Print evaluation results in a readable format."""
    print("\n" + "="*80)
    print("RAG EVALUATION RESULTS")
    print("="*80 + "\n")
    
    # Calculate averages
    metrics = ["faithfulness", "answer_relevance", "context_precision", "citation_quality"]
    avg_scores = {metric: [] for metric in metrics}
    
    for result in results:
        if "error" in result:
            continue
        for metric in metrics:
            if metric in result["scores"] and "score" in result["scores"][metric]:
                avg_scores[metric].append(result["scores"][metric]["score"])
    
    # Print summary
    print("SUMMARY SCORES (1-5 scale):")
    print("-" * 80)
    for metric in metrics:
        if avg_scores[metric]:
            avg = np.mean(avg_scores[metric])
            print(f"{metric.replace('_', ' ').title():25} {avg:.2f} / 5.00")
        else:
            print(f"{metric.replace('_', ' ').title():25} N/A")
    
    overall = np.mean([score for scores in avg_scores.values() for score in scores])
    print(f"\n{'OVERALL SCORE':25} {overall:.2f} / 5.00")
    print("="*80 + "\n")
    
    # Print individual results
    for i, result in enumerate(results, 1):
        print(f"\n{'='*80}")
        print(f"TEST CASE #{i}")
        print('='*80)
        print(f"\nQuestion: {result['question']}")
        print(f"\nAnswer: {result['answer']}")
        
        if result.get('expected_answer'):
            print(f"\nExpected: {result['expected_answer']}")
        
        print(f"\nRetrieved Contexts: {result.get('retrieved_contexts', 'N/A')}")
        
        if "error" not in result:
            print("\nScores:")
            for metric in metrics:
                if metric in result["scores"] and "score" in result["scores"][metric]:
                    score_data = result["scores"][metric]
                    print(f"  {metric.replace('_', ' ').title():25} {score_data['score']}/5 - {score_data.get('reasoning', '')}")
                    
                    # Print additional details
                    if metric == "faithfulness" and score_data.get("unsupported_claims"):
                        print(f"    Unsupported claims: {', '.join(score_data['unsupported_claims'])}")
                    elif metric == "citation_quality":
                        print(f"    Citations used: {score_data.get('citation_count', 0)}")
        else:
            print(f"\nError: {result['error']}")
    
    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG system quality")
    parser.add_argument("--domain", type=str, required=True, help="Domain to evaluate")
    parser.add_argument("--test-file", type=str, help="Path to test questions JSON file")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode - enter questions manually")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    args = parser.parse_args()
    
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY missing. Put it in your .env file.")
    
    client = OpenAI()
    
    # Load test cases
    test_cases = []
    
    if args.interactive:
        print("Interactive Evaluation Mode")
        print("Enter questions (empty line to finish):\n")
        while True:
            question = input("Question: ").strip()
            if not question:
                break
            test_cases.append({"question": question})
    elif args.test_file:
        test_file = Path(args.test_file)
        if not test_file.exists():
            raise FileNotFoundError(f"Test file not found: {test_file}")
        
        test_cases = json.loads(test_file.read_text(encoding="utf-8"))
        logger.info(f"Loaded {len(test_cases)} test cases from {test_file}")
    else:
        raise ValueError("Must specify either --test-file or --interactive")
    
    if not test_cases:
        print("No test cases provided.")
        return
    
    # Run evaluation
    results = []
    for test_case in test_cases:
        result = evaluate_single_question(client, args.domain, test_case)
        results.append(result)
    
    # Print results
    print_results(results)
    
    # Save results if output specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "domain": args.domain,
            "test_count": len(test_cases),
            "results": results,
        }
        
        output_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
