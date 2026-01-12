"""
Test and visualize chunking strategies for different document types.
Shows how different content is chunked and compares strategies.
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict
from chunk_pages import chunk_by_type, chunk_fixed_size, chunk_sentence_aware


def test_domain_chunking(domain: str, show_samples: int = 3):
    """
    Analyze chunking results for a domain.
    Shows statistics and sample chunks for each document type.
    """
    if Path.cwd().name == "src":
        base = Path("..")
    else:
        base = Path(".")
    
    pages_path = base / "data" / domain / "pages.jsonl"
    chunks_path = base / "data" / domain / "chunks.jsonl"
    
    if not pages_path.exists():
        print(f"❌ Pages file not found: {pages_path}")
        return
    
    if not chunks_path.exists():
        print(f"❌ Chunks file not found: {chunks_path}")
        print(f"💡 Run: python src/chunk_pages.py --domain {domain}")
        return
    
    print(f"\n{'='*80}")
    print(f"📊 CHUNKING ANALYSIS: {domain}")
    print(f"{'='*80}\n")
    
    # Analyze by document type
    type_stats = defaultdict(lambda: {
        'pages': 0,
        'chunks': 0,
        'total_chars': 0,
        'chunk_sizes': [],
        'samples': []
    })
    
    # Read chunks
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            source_type = chunk.get('source_type', 'unknown')
            text = chunk['text']
            
            stats = type_stats[source_type]
            stats['chunks'] += 1
            stats['total_chars'] += len(text)
            stats['chunk_sizes'].append(len(text))
            
            # Keep sample chunks
            if len(stats['samples']) < show_samples:
                stats['samples'].append({
                    'chunk_id': chunk['chunk_id'],
                    'text': text[:200] + '...' if len(text) > 200 else text,
                    'size': len(text)
                })
    
    # Read pages for comparison
    pages_by_type = defaultdict(int)
    with pages_path.open("r", encoding="utf-8") as f:
        for line in f:
            page = json.loads(line)
            source_type = page.get('source_type', 'unknown')
            pages_by_type[source_type] += 1
    
    # Update page counts
    for source_type, count in pages_by_type.items():
        type_stats[source_type]['pages'] = count
    
    # Display results by type
    for source_type in sorted(type_stats.keys()):
        stats = type_stats[source_type]
        
        print(f"\n📄 Document Type: {source_type.upper()}")
        print(f"   {'─'*70}")
        print(f"   Input records:  {stats['pages']}")
        print(f"   Output chunks:  {stats['chunks']}")
        print(f"   Ratio:          {stats['chunks']/stats['pages']:.2f}x")
        
        if stats['chunk_sizes']:
            avg_size = sum(stats['chunk_sizes']) / len(stats['chunk_sizes'])
            min_size = min(stats['chunk_sizes'])
            max_size = max(stats['chunk_sizes'])
            
            print(f"   Avg chunk size: {avg_size:.0f} chars")
            print(f"   Size range:     {min_size} - {max_size} chars")
        
        # Show sample chunks
        if stats['samples']:
            print(f"\n   📋 Sample chunks:")
            for i, sample in enumerate(stats['samples'], 1):
                print(f"\n      [{i}] {sample['chunk_id']} ({sample['size']} chars)")
                print(f"      {sample['text']}")
    
    # Overall summary
    total_pages = sum(s['pages'] for s in type_stats.values())
    total_chunks = sum(s['chunks'] for s in type_stats.values())
    
    print(f"\n{'='*80}")
    print(f"📈 SUMMARY")
    print(f"{'='*80}")
    print(f"   Total input records:  {total_pages}")
    print(f"   Total output chunks:  {total_chunks}")
    print(f"   Overall ratio:        {total_chunks/total_pages if total_pages > 0 else 0:.2f}x")
    print(f"   Document types:       {len(type_stats)}")
    print()


def compare_strategies(text: str, source_type: str = "txt"):
    """
    Compare different chunking strategies on the same text.
    Useful for understanding the differences.
    """
    print(f"\n{'='*80}")
    print(f"🔬 STRATEGY COMPARISON")
    print(f"{'='*80}")
    print(f"Text length: {len(text)} characters")
    print(f"Document type: {source_type}")
    print(f"\nFirst 300 chars of text:")
    print(f"{text[:300]}...\n")
    
    # Strategy 1: Fixed-size (old method)
    chunks_fixed = chunk_fixed_size(text, 2000, 300)
    print(f"\n1️⃣  FIXED-SIZE CHUNKING (old method)")
    print(f"   Chunks created: {len(chunks_fixed)}")
    for i, chunk in enumerate(chunks_fixed[:3], 1):
        print(f"\n   Chunk {i} ({len(chunk)} chars):")
        print(f"   Start: {chunk[:100]}...")
        print(f"   End:   ...{chunk[-100:]}")
    
    # Strategy 2: Sentence-aware
    chunks_sentence = chunk_sentence_aware(text, 2000, 300)
    print(f"\n2️⃣  SENTENCE-AWARE CHUNKING (new method)")
    print(f"   Chunks created: {len(chunks_sentence)}")
    for i, chunk in enumerate(chunks_sentence[:3], 1):
        print(f"\n   Chunk {i} ({len(chunk)} chars):")
        print(f"   Start: {chunk[:100]}...")
        print(f"   End:   ...{chunk[-100:]}")
    
    # Strategy 3: Document-type aware
    chunks_typed = chunk_by_type(text, source_type, has_structure=False)
    print(f"\n3️⃣  DOCUMENT-TYPE AWARE (automatic)")
    print(f"   Chunks created: {len(chunks_typed)}")
    for i, chunk in enumerate(chunks_typed[:3], 1):
        print(f"\n   Chunk {i} ({len(chunk)} chars):")
        print(f"   Start: {chunk[:100]}...")
        print(f"   End:   ...{chunk[-100:]}")
    
    print(f"\n{'='*80}\n")


def test_sample_texts():
    """Test chunking with various sample texts."""
    
    # Test 1: Structured text
    structured_text = """Introduction to Machine Learning

Machine learning is a subset of artificial intelligence. It focuses on building systems that learn from data.

Types of Machine Learning

There are three main types of machine learning. Supervised learning uses labeled data. Unsupervised learning finds patterns in unlabeled data. Reinforcement learning learns through trial and error.

Applications

Machine learning has many applications. These include image recognition, natural language processing, and recommendation systems. The field continues to grow rapidly."""
    
    print("\n" + "="*80)
    print("TEST 1: Structured Markdown-like Text")
    print("="*80)
    compare_strategies(structured_text, "md")
    
    # Test 2: Conversational audio transcript
    audio_text = """So today we're going to talk about something really interesting. You know I've been thinking about this for a while and I think it's important. The main thing is that we need to understand the fundamentals first. Without that foundation everything else becomes really difficult. Let me give you an example. When I was learning this myself I made a lot of mistakes. But those mistakes actually taught me valuable lessons. The key insight here is that practice matters more than theory in many cases. You can read all the books you want but until you actually try it yourself you won't really get it. Does that make sense? Okay so moving on to the next point."""
    
    print("\n" + "="*80)
    print("TEST 2: Conversational Audio Transcript")
    print("="*80)
    compare_strategies(audio_text, "audio")
    
    # Test 3: Image description
    image_text = """This image shows a complex neural network architecture diagram. The diagram contains multiple layers including input layer with 784 nodes, three hidden layers with 256, 128, and 64 nodes respectively, and an output layer with 10 nodes. Arrows connect all nodes between consecutive layers indicating fully connected layers. The activation functions are labeled as ReLU for hidden layers and Softmax for output layer."""
    
    print("\n" + "="*80)
    print("TEST 3: Image Description (should stay as single chunk)")
    print("="*80)
    compare_strategies(image_text, "jpg")


def main():
    parser = argparse.ArgumentParser(description="Test chunking strategies")
    parser.add_argument("--domain", type=str, help="Analyze chunking for a specific domain")
    parser.add_argument("--samples", type=int, default=3, help="Number of sample chunks to show per type")
    parser.add_argument("--test", action="store_true", help="Run sample text tests")
    parser.add_argument("--compare", type=str, help="Compare strategies on text from a file")
    parser.add_argument("--type", type=str, default="txt", help="Document type for comparison")
    
    args = parser.parse_args()
    
    if args.test:
        test_sample_texts()
    elif args.domain:
        test_domain_chunking(args.domain, args.samples)
    elif args.compare:
        file_path = Path(args.compare)
        if file_path.exists():
            text = file_path.read_text(encoding="utf-8")
            compare_strategies(text, args.type)
        else:
            print(f"❌ File not found: {file_path}")
    else:
        print("Usage:")
        print("  python src/test_chunking.py --domain <domain_name>")
        print("  python src/test_chunking.py --test")
        print("  python src/test_chunking.py --compare <file_path> --type pdf")
        print("\nExamples:")
        print("  python src/test_chunking.py --domain astronomy")
        print("  python src/test_chunking.py --test")


if __name__ == "__main__":
    main()
