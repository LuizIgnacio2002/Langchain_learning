from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

# Part 1: Create a ChatPromptTemplate using a template string
template = "Tell me a joke about {topic}"
prompt_template = ChatPromptTemplate.from_template(template)

print("Prompt template:")
prompt = prompt_template.invoke({"topic": "cats"})
print(prompt)
