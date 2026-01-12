"""Quick test to see chunking in action"""
from chunk_pages import chunk_by_type, chunk_fixed_size

# Sample long text that would get split
long_text = """The James Webb Space Telescope is amazing. It launched in December 2021. The telescope is designed to observe the universe in infrared light. This allows it to see through cosmic dust clouds. It can observe the first galaxies formed after the Big Bang. The telescope has four main instruments. These include NIRCam, NIRSpec, MIRI, and FGS/NIRISS. Each instrument serves a specific purpose. NIRCam is the primary imager. NIRSpec performs spectroscopy. MIRI observes mid-infrared wavelengths. The telescope orbits at the L2 Lagrange point. This is about 1.5 million kilometers from Earth. The position provides a stable environment. It also allows continuous observation of deep space. The sunshield keeps the instruments extremely cold. This is necessary for infrared observations. The telescope has already made groundbreaking discoveries. It has captured stunning images of distant galaxies. Scientists are using it to study exoplanet atmospheres. The data is revealing new insights about the universe. Future observations will continue to push the boundaries of astronomy."""

print("="*80)
print("BEFORE: Old fixed-size chunking")
print("="*80)
old_chunks = chunk_fixed_size(long_text, chunk_size=400, overlap=50)
for i, chunk in enumerate(old_chunks, 1):
    print(f"\n[Chunk {i}] {len(chunk)} chars")
    print(f"Starts: {chunk[:80]}")
    print(f"Ends:   {chunk[-80:]}")

print("\n\n" + "="*80)
print("AFTER: New sentence-aware chunking")
print("="*80)
new_chunks = chunk_by_type(long_text, source_type="txt", has_structure=False)
for i, chunk in enumerate(new_chunks, 1):
    print(f"\n[Chunk {i}] {len(chunk)} chars")
    print(f"Starts: {chunk[:80]}")
    print(f"Ends:   {chunk[-80:]}")

print("\n\n" + "="*80)
print("NOTICE THE DIFFERENCE:")
print("="*80)
print("[X] Old method: Splits at character 400, might break mid-sentence")
print("[+] New method: Splits at sentence boundaries for better context")
print(f"\nOld: {len(old_chunks)} chunks | New: {len(new_chunks)} chunks")
