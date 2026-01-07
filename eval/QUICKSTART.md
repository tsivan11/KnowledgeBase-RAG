# Quick Start: Evaluate Your RAG System

## 1. Interactive Mode (Test a few questions quickly)

```bash
python src/evaluate.py --domain astronomy --interactive
```

Enter questions when prompted:
```
Question: What is the James Webb Space Telescope?
Question: How does it differ from Hubble?
Question: [press Enter to finish]
```

## 2. Batch Mode (Run systematic tests)

First, create your test file (or use the sample):

**eval/my_tests.json:**
```json
[
  {
    "question": "What is the main mission of JWST?",
    "expected_answer": "Study early universe and exoplanets"
  },
  {
    "question": "What powers the Curiosity rover?",
    "expected_answer": "RTG (radioisotope thermoelectric generator)"
  }
]
```

Then run:
```bash
python src/evaluate.py --domain astronomy --test-file eval/my_tests.json
```

## 3. Save Results for Comparison

```bash
python src/evaluate.py --domain astronomy \
  --test-file eval/my_tests.json \
  --output eval/results_baseline.json
```

## What You'll See

```
RAG EVALUATION RESULTS
================================================================================

SUMMARY SCORES (1-5 scale):
--------------------------------------------------------------------------------
Faithfulness               4.80 / 5.00
Answer Relevance           4.60 / 5.00
Context Precision          4.40 / 5.00
Citation Quality           5.00 / 5.00

OVERALL SCORE              4.70 / 5.00
================================================================================

TEST CASE #1
================================================================================

Question: What is the main mission of JWST?

Answer: The James Webb Space Telescope's main mission is to study the early 
universe, observe the formation of the first galaxies, and analyze the 
atmospheres of exoplanets [1][2].

Scores:
  Faithfulness               5/5 - All claims supported by context
  Answer Relevance           5/5 - Directly answers the question
  Context Precision          4/5 - Most contexts are relevant
  Citation Quality           5/5 - Good citation coverage: 2 sources cited
    Citations used: 2
```

## Interpreting Results

- **4.5-5.0**: Excellent - Production ready
- **3.5-4.4**: Good - Minor improvements needed
- **2.5-3.4**: Fair - Needs work on prompts or retrieval
- **< 2.5**: Poor - Major issues with RAG pipeline

## Common Issues and Fixes

**Low Faithfulness (< 3.5)**
→ System is hallucinating
→ Fix: Make prompts stricter, use temperature=0

**Low Context Precision (< 3.5)**
→ Retrieval is pulling irrelevant chunks
→ Fix: Adjust chunk size, improve chunking strategy

**Low Citation Quality (< 3.5)**
→ Not citing sources properly
→ Fix: Improve system prompt to require citations

## Next Steps

1. Run evaluation on your domain
2. Identify weak areas from the scores
3. Make improvements to prompts/chunking/retrieval
4. Re-run evaluation to verify improvements
5. Save results to track progress over time

See [eval/README.md](README.md) for detailed documentation.
