import os
from google import genai
from dotenv import load_dotenv
from src.config.config import GOOGLE_API_KEY

# Initialize the client (it automatically looks for GOOGLE_API_KEY)
if GOOGLE_API_KEY == None:
    raise KeyError("Gemini Key not present!")

client = genai.Client(api_key=GOOGLE_API_KEY)

def ask_gemini(client, prompt):
    response = client.models.generate_content(
        model="gemini-2.5-flash", 
        contents=prompt
    )
    return response.text

# Test it out
if __name__ == "__main__":
    result = ask_gemini("Tell me a short joke about a data scientist.")
    print(f"Gemini: {result}")