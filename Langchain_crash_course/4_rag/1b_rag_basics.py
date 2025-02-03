import os

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__)) # abspath returns the absolute path of the file | __file__ is the path of the current file
persistent_directory = os.path.join(current_dir, "db", "chroma_db") # join combines the current directory with the path to the database directory

# Define the embedding model
embedding = OpenAIEmbeddings(model="text-embedding-3-small") # create an OpenAIEmbeddings object with the model name

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embedding) # create a Chroma object with the persistent directory and embedding function

# Define the user's question
query = "How does Telemachus’ voyage to Pylos affect the story of Penelope and the suitors?" # define the user's question

# Retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity_score_threshold",  # define the search type
    search_kwargs={"k": 2, "score_threshold": 0.20}  # Ensure 'score_threshold' is a float value between 0 and 1
)

relevant_docs = retriever.invoke(query) # retrieve relevant documents based on the query

print("Relevant documents:")
for i, doc in enumerate(relevant_docs):
    print(f"Document {i + 1}: {doc.page_content}") 
    print("-" * 50) 
    if doc.metadata:
        print(f"Metadata: {doc.metadata}")
    print("=" * 50)
    print("\n")