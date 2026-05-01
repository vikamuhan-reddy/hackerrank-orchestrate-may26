import os
import sys
import pandas as pd
import json
import hashlib
import asyncio
import argparse
from dotenv import load_dotenv
from openai import AsyncOpenAI
import vector_store

load_dotenv()
client = AsyncOpenAI(
    api_key=(os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")),
    base_url="https://api.groq.com/openai/v1"
)
CACHE_FILE = "cache.json"

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            try: return json.load(f)
            except: return {}
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)

async def reformulate_query(query):
    try:
        completion = await client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a search expert. Rewrite the user's support ticket query into 3-5 keywords separated by spaces to maximize BM25 similarity."},
                {"role": "user", "content": f"Query: {query}"}
            ],
            temperature=0.0,
            seed=42,
            max_tokens=40
        )
        return completion.choices[0].message.content.strip()
    except: return query

async def predict_ticket(ticket_issue, semaphore, debug=False):
    async with semaphore:
        docs_with_scores = vector_store.search(ticket_issue, limit=3)
        top_score = docs_with_scores[0]['score'] if docs_with_scores else 0
        expanded_query = ""

        if top_score < 1.0:
            expanded_query = await reformulate_query(ticket_issue)
            reformulated_docs = vector_store.search(expanded_query, limit=3)
            ref_score = reformulated_docs[0]['score'] if reformulated_docs else 0
            if ref_score > top_score:
                docs_with_scores = reformulated_docs
                top_score = ref_score

        confidence = min(0.98, top_score / 30.0) if top_score > 0 else 0.0
        
        # 1. Dynamic Thresholding (DeepSeek Recommendation)
        # Critical billing/order issues require higher confidence (0.7) than standard product bugs (0.5)
        # We start with a generic low-pass and refine after LLM classification if needed.
        if top_score < 0.8:
            return json.dumps({
                "status": "escalated",
                "product_area": "Unknown",
                "response": "Escalated to human support.",
                "justification": "Confidence threshold not met. Keyword matching yielded insufficient corpus context.",
                "request_type": "support_request",
                "confidence": round(confidence, 2),
                "escalation_reason": "Low BM25 Retrieval Score",
                "suggested_search": expanded_query
            })

        docs = [d['chunk'] for d in docs_with_scores]
        context = "\n\n".join([f"Source: {d['source']} ({d['company']})\n{d['text']}" for d in docs])
        
        prompt = f"""You are a support agent triage assistant for HackerRank, Claude, and Visa.
        
<CORPUS>
{context}
</CORPUS>

Ticket Issue:
{ticket_issue}

Respond strictly in JSON with these keys: 
"status", "product_area", "response", "justification", "request_type".

Decision Logic: 
- If information is missing or sensitive (refunds/bans), status MUST be 'escalated'.
- Justification MUST cite specific Source files.
"""

        completion = await client.chat.completions.create(
            model="llama-3.1-8b-instant",
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": "You are a professional support agent. Provide structured triage decisions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            seed=42
        )
        
        try:
            res_dict = json.loads(completion.choices[0].message.content)
            
            # 2. Advanced Escalation Sweep (DeepSeek): Tighter threshold for Billing
            is_critical = any(kw in str(res_dict).lower() for kw in ["billing", "refund", "payment", "card", "charge"])
            if is_critical and confidence < 0.7:
                 res_dict["status"] = "escalated"
                 res_dict["justification"] = f"(Safety Threshold) Escorted billing query to human due to moderate confidence score ({confidence:.2f})."
            
            res_dict["confidence"] = round(confidence, 2)
            res_dict["suggested_search"] = expanded_query
            return json.dumps(res_dict)
        except: return completion.choices[0].message.content

async def process_all_tickets(debug=False):
    if not os.path.exists("embeddings.pkl"):
        import parser
        chunks = parser.load_all_chunks(data_dir="../data")
        vector_store.build_index(chunks, output_path="embeddings.pkl")

    df = pd.read_csv("../support_tickets/support_tickets.csv")
    print(f"Processing {len(df)} tickets (Async Concurrency=2)...")
    
    cache = load_cache()
    semaphore = asyncio.Semaphore(2)
    tasks = []
    
    async def process_row(i, issue):
        ih = hashlib.md5(issue.encode()).hexdigest()
        if ih in cache: return i, cache[ih]
        for _ in range(3):
            try:
                res_str = await predict_ticket(issue, semaphore, debug=debug)
                res_dict = json.loads(res_str)
                cache[ih] = res_dict
                return i, res_dict
            except Exception as e:
                if '429' in str(e): await asyncio.sleep(40)
                else: return i, {"status":"escalated","product_area":"Error","response":"Error","justification":str(e),"request_type":"support_request"}

    results_list = await asyncio.gather(*[process_row(i, str(r.get('Issue',''))) for i,r in df.iterrows()])
    save_cache(cache)
    
    results_list.sort(key=lambda x: x[0])
    
    # 3. COMPLIANCE CHECK: Only the 5 standard columns for the output.csv
    final_output = []
    required_cols = ["status", "product_area", "response", "justification", "request_type"]
    for _, res in results_list:
        clean_res = {k: res.get(k, "") for k in required_cols}
        final_output.append(clean_res)
        
    pd.DataFrame(final_output).to_csv("../support_tickets/output.csv", index=False)
    print(f"Complete. Saved compliant predictions to ../support_tickets/output.csv")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--debug", action="store_true")
    asyncio.run(process_all_tickets(debug=p.parse_args().debug))

if __name__ == "__main__":
    main()
