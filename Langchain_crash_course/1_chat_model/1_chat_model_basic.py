from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model_name="gpt-3.5-turbo")

result = model.invoke("What is the capital of India?")
print("Full result")
print(result)
print("Content only")
print(result.content)