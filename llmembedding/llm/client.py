from google import genai
from dotenv import load_dotenv
import os

loaded=load_dotenv()

class Client:
    def __init__(self):
        self.client=genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def generate(self, prompt):
        response=self.client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
        )
        
        return response.text