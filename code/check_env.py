import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

def main():
    print("--- HackerRank Orchestrate: Cold Start Validation ---")
    
    # 1. Check directories
    if not os.path.exists("../data"):
        print("❌ Error: '../data' corpus directory not found.")
        sys.exit(1)
    print("✅ Found '../data' corpus directory.")
    
    if not os.path.exists("../support_tickets/support_tickets.csv"):
        print("❌ Error: '../support_tickets/support_tickets.csv' not found.")
        sys.exit(1)
    print("✅ Found '../support_tickets/support_tickets.csv'.")
    
    # 2. Check .env
    env_path = "../.env"
    if not os.path.exists(env_path):
        env_path = ".env"
    
    load_dotenv(dotenv_path=env_path)
    api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: Valid key missing from .env.")
        sys.exit(1)
    if "your_real_key" in api_key or api_key.startswith("sk-5678"):
        print("⚠️ Warning: Key looks like a placeholder. Please ensure it is valid.")
    else:
        print("✅ API_KEY loaded.")
    
    # 3. Test Groq Connection
    print("⚙️ Testing Groq API connection...")
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Hello! Reply strictly with the phrase 'API Connection OK'."}],
            max_tokens=20
        )
        response = completion.choices[0].message.content.strip()
        print(f"✅ Groq API Test Successful: '{response}'")
    except Exception as e:
        print(f"❌ Error connecting to Groq API: {e}")
        sys.exit(1)
        
    print("\n🟢 Environment is fully validated and ready for execution! You may now run `python main.py`.")
    
if __name__ == "__main__":
    main()
