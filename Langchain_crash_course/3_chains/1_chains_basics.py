from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo")

# define prompt templates (no need for seprarate runnable chains)
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a comedian who tells jokes about {topic}"),
        ("human", "tell me {joke_count} jokes "),
    ]
)

# Create the combined chain using LangChain Expression Language (LCEL)
chain = prompt_template | model | StrOutputParser() # | is the pipe operator used to combine chains
# StrOutputParser is used to parse the output of the model as a string

result = chain.invoke({"topic": "cats", "joke_count": "3"})

print("Full result")
print(result)