import os
import pickle
from rank_bm25 import BM25Okapi

def build_index(chunks, output_path="embeddings.pkl"):
    print("Building BM25 index locally...")
    tokenized_corpus = [c['text'].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    
    with open(output_path, 'wb') as f:
        pickle.dump({
            'bm25': bm25,
            'metadata': chunks
        }, f)
    print(f"Saved to {output_path}")

def load_index(index_path="embeddings.pkl"):
    with open(index_path, 'rb') as f:
        return pickle.load(f)

def search(query, limit=3, index_path="embeddings.pkl"):
    data = load_index(index_path)
    bm25 = data['bm25']
    metadata = data['metadata']
    
    tokenized_query = query.lower().split()
    doc_scores = bm25.get_scores(tokenized_query)
    
    # get top N indices
    top_indices = doc_scores.argsort()[::-1][:limit]
    top_scores = doc_scores[top_indices]
    
    return [{"chunk": metadata[i], "score": top_scores[idx]} for idx, i in enumerate(top_indices)]

if __name__ == "__main__":
    from parser import load_all_chunks
    chunks = load_all_chunks()
    build_index(chunks)
