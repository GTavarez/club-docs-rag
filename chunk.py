from pathlib import Path

CHUNK_SIZE = 800

def chunk_text(text, size=CHUNK_SIZE):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) < size:
            current += p + "\n\n"
        else:
            if current:
                chunks.append(current.strip())
            current = p + "\n\n"
    if current:
        chunks.append(current.strip())
    return chunks


all_chunks = []
for path in Path("docs").glob("*.txt"):
    text = path.read_text(encoding="utf-8")
    for c in chunk_text(text):
        all_chunks.append({"source": path.name, "text": c})

print(f"{len(all_chunks)} chunks from {len(list(Path('docs').glob('*.txt')))} files\n")
for c in all_chunks[:3]:
    print("=" * 60)
    print(c["source"])
    print(c["text"][:300])