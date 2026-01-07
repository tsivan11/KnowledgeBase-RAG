# RAG Evaluation Guide

## Overview

The evaluation system measures your RAG quality across 4 key metrics:

1. **Faithfulness** (1-5): Does the answer stick to the retrieved context without hallucinating?
2. **Answer Relevance** (1-5): Does the answer actually address the question?
3. **Context Precision** (1-5): Are the retrieved chunks relevant to the question?
4. **Citation Quality** (1-5): Are sources properly cited?

## Quick Start

### 1. Interactive Mode (Quick Testing)

Test individual questions on the fly:

```bash
python src/evaluate.py --domain astronomy --interactive
```

Then enter your questions one by one. Press Enter on an empty line when done.

### 2. Batch Mode (Systematic Testing)

Create or edit a test file, then run:

```bash
python src/evaluate.py --domain astronomy --test-file eval/sample_test_questions.json
```

Save results to a file:

```bash
python src/evaluate.py --domain astronomy --test-file eval/sample_test_questions.json --output eval/results_2026-01-07.json
```

## Creating Test Datasets

Test files are JSON arrays with this structure:

```json
[
  {
    "question": "Your question here",
    "expected_answer": "Optional - what you expect the answer to be"
  },
  {
    "question": "Another question",
    "expected_answer": null,
    "notes": "Optional - notes about this test case"
  }
]
```

### Tips for Good Test Cases

**Coverage**: Test different question types:
- Factual ("What is X?")
- Comparative ("What's the difference between X and Y?")
- Explanatory ("How does X work?")
- Out-of-scope ("What is the population of Jupiter?") - should say "I don't know"

**Domain-Specific**: Create test sets for each domain:
- `eval/astronomy_tests.json`
- `eval/contracts_tests.json`
- `eval/computer_vision_tests.json`

**Progressive Difficulty**:
- Easy: Questions with direct answers in one document
- Medium: Questions requiring multiple sources
- Hard: Questions needing synthesis across documents

## Understanding Results

### Sample Output

```
SUMMARY SCORES (1-5 scale):
---------------------------------------------------
Faithfulness               4.80 / 5.00
Answer Relevance           4.60 / 5.00
Context Precision          4.40 / 5.00
Citation Quality           5.00 / 5.00

OVERALL SCORE              4.70 / 5.00
```

### What the Scores Mean

**5/5 - Excellent**
- Faithfulness: No hallucinations, all claims supported
- Relevance: Directly and fully answers the question
- Precision: All retrieved chunks are relevant
- Citations: Multiple proper citations

**3-4/5 - Good**
- Minor issues, generally acceptable
- May have slight irrelevancies or missing citations

**1-2/5 - Needs Improvement**
- Hallucinations, irrelevant content, or no citations
- Indicates problems with retrieval or generation

## Troubleshooting Low Scores

### Low Faithfulness (< 3.0)
**Problem**: System is hallucinating or adding unsupported information

**Solutions**:
- Make your system prompt stricter
- Increase temperature=0 for deterministic outputs
- Improve chunking to provide better context
- Use a better generation model (e.g., upgrade from gpt-4o-mini to gpt-4o)

### Low Answer Relevance (< 3.0)
**Problem**: Answers don't address the questions

**Solutions**:
- Improve embedding model or query reformulation
- Check if questions are out of domain scope
- Review prompt instructions

### Low Context Precision (< 3.0)
**Problem**: Retrieved chunks aren't relevant

**Solutions**:
- Adjust chunk size (try 400-800 tokens)
- Improve chunk overlap
- Try different embedding models
- Add metadata filters
- Increase/decrease TOP_K parameter

### Low Citation Quality (< 3.0)
**Problem**: Sources not properly cited

**Solutions**:
- Make citation requirements more explicit in prompt
- Provide examples in the system prompt
- Check if context blocks are properly numbered

## Best Practices

### Regression Testing

Run evaluations after every change to:
- Chunking strategy
- Embedding model
- Retrieval parameters (TOP_K)
- System prompts
- LLM model selection

Save results with timestamps:
```bash
python src/evaluate.py --domain astronomy \
  --test-file eval/astronomy_tests.json \
  --output eval/results_$(date +%Y%m%d).json
```

### Continuous Monitoring

1. **Build a golden dataset** (20-50 questions per domain)
2. **Run weekly evaluations** to catch quality degradation
3. **Track scores over time** in a spreadsheet
4. **Set quality thresholds** (e.g., "Faithfulness must be > 4.0")

### Domain-Specific Testing

Each domain should have its own test file:

```
eval/
  ├── astronomy_tests.json
  ├── contracts_tests.json
  ├── computer_vision_tests.json
  └── sample_test_questions.json
```

Run domain-specific evaluations:
```bash
python src/evaluate.py --domain astronomy --test-file eval/astronomy_tests.json
```

## Advanced: Using RAGAS Framework (Optional)

For more sophisticated metrics, you can use the RAGAS library:

1. Install: `pip install ragas`
2. RAGAS provides additional metrics like:
   - Context Recall
   - Context Utilization
   - Answer Semantic Similarity

The current evaluation script uses LLM-as-judge which is simpler and doesn't require ground truth labels.

## Example Workflow

### Initial Setup
```bash
# 1. Create test questions for your domain
cp eval/sample_test_questions.json eval/my_domain_tests.json
# Edit the file with domain-specific questions

# 2. Run baseline evaluation
python src/evaluate.py --domain my_domain \
  --test-file eval/my_domain_tests.json \
  --output eval/baseline.json
```

### After Making Changes
```bash
# Run evaluation to see if quality improved
python src/evaluate.py --domain my_domain \
  --test-file eval/my_domain_tests.json \
  --output eval/after_changes.json

# Compare results manually or with a script
```

### Before Production Deploy
```bash
# Run full test suite
for domain in astronomy contracts coursework; do
  python src/evaluate.py --domain $domain \
    --test-file eval/${domain}_tests.json \
    --output eval/${domain}_prod_check.json
done
```

## Interpreting Edge Cases

**"I don't know" Answers**:
- High faithfulness (good - not hallucinating)
- Low relevance if the answer IS in the docs (retrieval problem)
- High relevance if the answer ISN'T in the docs (correct behavior)

**Over-citing** (e.g., "[1][2][3][4][5]"):
- High citation score
- May indicate retrieval is good
- Could be verbose - review if necessary

**No Citations**:
- Low citation score
- Check if system prompt is clear about citing sources
- Verify answer isn't legitimately saying "I don't know"

---

## Questions?

The evaluation script uses GPT-4o-mini as the judge by default. For more accurate evaluations, you can upgrade to GPT-4 by changing `JUDGE_MODEL` in [src/evaluate.py](../src/evaluate.py#L25).
