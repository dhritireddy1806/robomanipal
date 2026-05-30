import os
from google import genai
client=genai.Client(api_key=os.environ["GEMINI_API_KEY"])
# in the terminal: export GEMINI_API_KEY="your_api_key_here"
sent=input("Enter your prompt: ")
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=sent
)

print(response.text)