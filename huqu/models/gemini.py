from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

class GeminiModel:
    "Gemini text to text model"
    
    def generate(self, prompt: str) -> str:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model='gemini-1.5-flash-8b', 
            contents=prompt
        )
        return response.text