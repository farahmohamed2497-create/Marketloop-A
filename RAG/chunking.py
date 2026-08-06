def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start += chunk_size - overlap

    return chunks


def chunk_document(text: str, base_metadata: dict, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    
    raw_chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    result = []

    for i, chunk in enumerate(raw_chunks):
        metadata = dict(base_metadata)
        metadata["chunk_index"] = i
        result.append({
            "text": chunk,
            "metadata": metadata
        })

    return result