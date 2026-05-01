import os
import glob

def get_markdown_files(data_dir="../data"):
    return glob.glob(os.path.join(data_dir, "**", "*.md"), recursive=True)

def chunk_markdown(filepath, overlap_lines=4):
    # Basic heading-based chunker
    chunks = []
    current_chunk = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("#"):
                if current_chunk:
                    chunks.append("\n".join(current_chunk).strip())
                    # Configurable sliding window parameter to maintain context flow intelligently
                    current_chunk = current_chunk[-overlap_lines:] if len(current_chunk) >= overlap_lines else current_chunk
            
            if line.strip():
                current_chunk.append(line.strip())
                
        if current_chunk:
            chunks.append("\n".join(current_chunk).strip())

    source = filepath.split('/')[-1]
    company = "None"
    if "/visa" in filepath.lower(): company = "Visa"
    elif "/claude" in filepath.lower(): company = "Claude"
    elif "/hackerrank" in filepath.lower(): company = "HackerRank"

    return [{"text": c, "source": source, "company": company} for c in chunks if len(c) > 50]

def load_all_chunks(data_dir="../data", overlap_lines=4):
    all_chunks = []
    files = get_markdown_files(data_dir)
    for f in files:
        all_chunks.extend(chunk_markdown(f, overlap_lines=overlap_lines))
    return all_chunks

if __name__ == "__main__":
    chunks = load_all_chunks()
    print(f"Parsed {len(chunks)} chunks from data directory.")
