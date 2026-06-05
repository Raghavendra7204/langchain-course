from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)
print(os.environ.get("GOOGLE_API_KEY"))
response = llm.invoke("Tell me a joke")

print(response.content)