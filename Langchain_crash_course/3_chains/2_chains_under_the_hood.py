from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}"),
        ("human", "Tell me {joke_count} jokes."),
    ]
)

# Fix: Pass dictionary directly to `invoke()`
format_prompt = RunnableLambda(lambda x: prompt_template.invoke(x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))   
parse_output = RunnableLambda(lambda x: x.content)

# Correct usage of RunnableSequence
chain = RunnableSequence(format_prompt, invoke_model, parse_output)

# Run the chain
result = chain.invoke({"topic": "cats", "joke_count": "3"})
print("Full result")
print(result)
