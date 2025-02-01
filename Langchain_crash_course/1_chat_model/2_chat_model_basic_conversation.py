from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo")

# SystemMessage: Message for priming AI behavior, usually used to start a conversation
# HumanMessage: Message from a human user
# AIMessage: Message from the AI model
message = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 2 + 2?"),
    AIMessage(content="2 + 2 = 4"),
    HumanMessage(content="What is 3 + 3?"),

]

# invoke() method is used to send a message to the AI model
result = model.invoke(message)
print("Full result")
print(result)
print("---------------------------")
print("Content only")
print(result.content)
