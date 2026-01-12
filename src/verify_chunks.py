"""
Verify chunking quality - checks for common issues and best practices.
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict
import re


def verify_chunks(domain: str):
    """Analyze chunk quality and detect potential issues."""
    
    if Path.cwd().name == "src":
        base = Path("..")
    else:
        base = Path(".")
    
    chunks_path = base / "data" / domain / "chunks.jsonl"
    
    if not chunks_path.exists():
        print(f"[!] Chunks file not found: {chunks_path}")
        print(f"    Run: python src/chunk_pages.py --domain {domain}")
        return
    
    print(f"\n{'='*80}")
    print(f"CHUNK QUALITY VERIFICATION: {domain}")
    print(f"{'='*80}\n")
    
    # Track issues by type
    issues = defaultdict(list)
    stats_by_type = defaultdict(lambda: {
        'count': 0,
        'total_chars': 0,
        'sizes': [],
        'mid_sentence_breaks': 0,
        'too_small': 0,
        'too_large': 0,
    })
    
    # Quality thresholds
    MIN_REASONABLE_SIZE = 50
    MAX_REASONABLE_SIZE = 5000
    
    chunks_by_source = defaultdict(list)
    
    with chunks_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                chunk = json.loads(line)
                source_type = chunk.get('source_type', 'unknown')
                text = chunk['text']
                chunk_id = chunk['chunk_id']
                source = chunk['source']
                
                # Group by source for overlap analysis
                chunks_by_source[source].append(chunk)
                
                stats = stats_by_type[source_type]
                stats['count'] += 1
                stats['total_chars'] += len(text)
                stats['sizes'].append(len(text))
                
                # Check 1: Mid-sentence breaks (for text-based documents)
                if source_type in ['txt', 'md', 'html', 'docx', 'pdf', 'audio']:
                    # Check if chunk ends mid-sentence
                    last_chars = text.rstrip()[-50:] if len(text) > 50 else text
                    if last_chars and not re.search(r'[.!?\n]$', last_chars.rstrip()):
                        # Could be mid-sentence (not definitive, but suspicious)
                        if len(text) > 100:  # Only flag if chunk is substantial
                            stats['mid_sentence_breaks'] += 1
                            if stats['mid_sentence_breaks'] <= 3:  # Sample first 3
                                issues['mid_sentence'].append({
                                    'chunk_id': chunk_id,
                                    'type': source_type,
                                    'ending': last_chars[-80:] if len(last_chars) > 80 else last_chars
                                })
                
                # Check 2: Unusually small chunks
                if len(text) < MIN_REASONABLE_SIZE:
                    stats['too_small'] += 1
                    if stats['too_small'] <= 3:
                        issues['too_small'].append({
                            'chunk_id': chunk_id,
                            'type': source_type,
                            'size': len(text),
                            'text': text[:100]
                        })
                
                # Check 3: Unusually large chunks (might need splitting)
                if len(text) > MAX_REASONABLE_SIZE:
                    stats['too_large'] += 1
                    if stats['too_large'] <= 3:
                        issues['too_large'].append({
                            'chunk_id': chunk_id,
                            'type': source_type,
                            'size': len(text)
                        })
                
            except Exception as e:
                issues['parsing_errors'].append({
                    'line': line_num,
                    'error': str(e)
                })
    
    # Print statistics by document type
    print("STATISTICS BY DOCUMENT TYPE")
    print("-" * 80)
    
    for source_type in sorted(stats_by_type.keys()):
        stats = stats_by_type[source_type]
        
        if stats['count'] == 0:
            continue
        
        avg_size = stats['total_chars'] / stats['count']
        min_size = min(stats['sizes'])
        max_size = max(stats['sizes'])
        
        print(f"\n{source_type.upper()}")
        print(f"  Chunks:         {stats['count']}")
        print(f"  Avg size:       {avg_size:.0f} chars")
        print(f"  Size range:     {min_size} - {max_size} chars")
        
        # Quality indicators
        quality_issues = []
        if stats['mid_sentence_breaks'] > 0:
            quality_issues.append(f"{stats['mid_sentence_breaks']} potential mid-sentence breaks")
        if stats['too_small'] > 0:
            quality_issues.append(f"{stats['too_small']} too small (<{MIN_REASONABLE_SIZE} chars)")
        if stats['too_large'] > 0:
            quality_issues.append(f"{stats['too_large']} too large (>{MAX_REASONABLE_SIZE} chars)")
        
        if quality_issues:
            print(f"  [!] Issues:     {', '.join(quality_issues)}")
        else:
            print(f"  [+] Quality:    Good")
    
    # Print detailed issues
    print(f"\n\n{'='*80}")
    print("DETAILED ISSUE REPORT")
    print(f"{'='*80}\n")
    
    total_issues = sum(len(v) for v in issues.values())
    
    if total_issues == 0:
        print("[+] No significant issues found!")
        print("    All chunks appear to be properly formatted.")
        return
    
    # Mid-sentence breaks
    if 'mid_sentence' in issues and issues['mid_sentence']:
        print(f"\n[!] POTENTIAL MID-SENTENCE BREAKS ({len(issues['mid_sentence'])} samples)")
        print("-" * 80)
        for issue in issues['mid_sentence'][:5]:
            print(f"\nChunk: {issue['chunk_id']}")
            print(f"Type:  {issue['type']}")
            print(f"Ends:  ...{issue['ending']}")
    
    # Too small chunks
    if 'too_small' in issues and issues['too_small']:
        print(f"\n\n[!] CHUNKS TOO SMALL (<{MIN_REASONABLE_SIZE} chars) ({len(issues['too_small'])} samples)")
        print("-" * 80)
        for issue in issues['too_small'][:5]:
            print(f"\nChunk: {issue['chunk_id']}")
            print(f"Type:  {issue['type']}")
            print(f"Size:  {issue['size']} chars")
            print(f"Text:  {issue['text']}")
    
    # Too large chunks
    if 'too_large' in issues and issues['too_large']:
        print(f"\n\n[!] CHUNKS TOO LARGE (>{MAX_REASONABLE_SIZE} chars) ({len(issues['too_large'])} samples)")
        print("-" * 80)
        for issue in issues['too_large'][:5]:
            print(f"\nChunk: {issue['chunk_id']}")
            print(f"Type:  {issue['type']}")
            print(f"Size:  {issue['size']} chars")
    
    # Parsing errors
    if 'parsing_errors' in issues and issues['parsing_errors']:
        print(f"\n\n[!] PARSING ERRORS ({len(issues['parsing_errors'])})")
        print("-" * 80)
        for issue in issues['parsing_errors'][:5]:
            print(f"Line {issue['line']}: {issue['error']}")
    
    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    total_chunks = sum(s['count'] for s in stats_by_type.values())
    print(f"Total chunks analyzed: {total_chunks}")
    print(f"Document types: {len(stats_by_type)}")
    print(f"Issues found: {total_issues}")
    
    if total_issues == 0:
        print("\n[+++] EXCELLENT! No issues detected.")
    elif total_issues < 10:
        print("\n[++] GOOD! Minor issues detected, likely acceptable.")
    elif total_issues < 50:
        print("\n[+] FAIR! Some issues detected, review recommended.")
    else:
        print("\n[!] NEEDS ATTENTION! Significant issues detected.")
    
    print()


def show_chunk_samples(domain: str, source_type: str = None, count: int = 3):
    """Show sample chunks for inspection."""
    
    if Path.cwd().name == "src":
        base = Path("..")
    else:
        base = Path(".")
    
    chunks_path = base / "data" / domain / "chunks.jsonl"
    
    if not chunks_path.exists():
        print(f"[!] Chunks file not found: {chunks_path}")
        return
    
    print(f"\n{'='*80}")
    print(f"CHUNK SAMPLES: {domain}")
    if source_type:
        print(f"Filtered by type: {source_type}")
    print(f"{'='*80}\n")
    
    shown = 0
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            
            if source_type and chunk.get('source_type') != source_type:
                continue
            
            print(f"\nChunk ID: {chunk['chunk_id']}")
            print(f"Type:     {chunk.get('source_type', 'unknown')}")
            print(f"Size:     {len(chunk['text'])} chars")
            print(f"Page:     {chunk.get('page', 'N/A')}")
            print(f"Section:  {chunk.get('section', 'N/A')}")
            print("-" * 80)
            print(chunk['text'][:500])
            if len(chunk['text']) > 500:
                print("\n[... truncated ...]")
            print()
            
            shown += 1
            if shown >= count:
                break
    
    if shown == 0:
        print(f"No chunks found" + (f" for type '{source_type}'" if source_type else ""))


def main():
    parser = argparse.ArgumentParser(description="Verify chunking quality")
    parser.add_argument("--domain", type=str, required=True, help="Domain to verify")
    parser.add_argument("--samples", action="store_true", help="Show sample chunks")
    parser.add_argument("--type", type=str, help="Filter by document type for samples")
    parser.add_argument("--count", type=int, default=3, help="Number of samples to show")
    
    args = parser.parse_args()
    
    if args.samples:
        show_chunk_samples(args.domain, args.type, args.count)
    else:
        verify_chunks(args.domain)


if __name__ == "__main__":
    main()
