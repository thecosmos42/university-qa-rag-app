from google import genai
from src.config.config import GOOGLE_API_KEY

client = genai.Client(api_key=GOOGLE_API_KEY)

def ask_gemini(client, prompt):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    usage = response.usage_metadata
    return {
        "text": response.text,
        "tokens": {
            "prompt": usage.prompt_token_count if usage else 0,
            "response": usage.candidates_token_count if usage else 0,
            "total": usage.total_token_count if usage else 0,
        }
    }