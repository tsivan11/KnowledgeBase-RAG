from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

print("Testing OpenAI API connection...")
print(f"API Key present: {bool(os.getenv('OPENAI_API_KEY'))}")

try:
    client = OpenAI()
    resp = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': 'Say hello'}]
    )
    print(f"✓ API works: {resp.choices[0].message.content}")
except Exception as e:
    print(f"✗ API failed: {e}")
