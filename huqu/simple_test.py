import os
import json
import requests
from datasets import load_dataset
import base64
from PIL import Image
from io import BytesIO

def encode_image(image: Image.Image) -> str:
    """Convert PIL Image to base64 string. Necessary for Groq API."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def main():
    # Load Groq API key from environment
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Please set GROQ_API_KEY environment variable")

    # Load the first image from a dataset
    dataset = load_dataset("keremberke/german-traffic-sign-detection", split="train")
    image = dataset[0]['image']  # This is already a PIL Image

    # Prepare the API request
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.2-11b-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What do you see in this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encode_image(image)}"
                        }
                    }
                ]
            }
        ]
    }

    # Make the request
    print("Sending request to Groq API...")
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload
    )

    # Save the response
    result = {
        "status_code": response.status_code,
        "response": response.json()
    }
    
    with open("groq_response.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Response saved to groq_response.json")

if __name__ == "__main__":
    main() 