import os
from openai import OpenAI
import config as config  # noqa: F401  (importing config loads .env before the key is read)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

# One OpenAI client shared by the brochure pipeline and the RAG code.
client = OpenAI(api_key=api_key)
