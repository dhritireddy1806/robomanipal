import os
from dotenv import load_dotenv
loaded=load_dotenv()  

from google import genai
client=genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

sent=input("Enter your prompt: ")
response = client.models.generate_content(
    model="gemini-2.5-flash",
   contents=sent
)

print(response.text)