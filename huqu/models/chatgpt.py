from openai import OpenAI
import os
from dotenv import load_dotenv
import base64
from io import BytesIO
from PIL import Image

load_dotenv()

class GPT4oMiniLLM():
    def __init__(self,model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content

class GPT4oMiniMLLM():
    def __init__(self,model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        
    def encode_image(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string.
        
        Args:
            image: PIL Image to encode
            
        Returns:
            Base64 encoded string of the image
        """
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def generate(self, image: Image.Image, prompt: str) -> str:
        base64_image = self.encode_image(image)
        response = self.client.chat.completions.create(
        model=self.model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            },
            ],
        )

        return response.choices[0].message.content
