import os

from langchain.text_splitter import CharacterTextSplitter # CharacterTextSplitter makes it easier to split text into smaller chunks
from langchain_community.document_loaders import TextLoader # TextLoader is a document loader that loads text from a file
from langchain_chroma import Chroma  # Update this import # Chroma is a vector store that stores vectors in a database
from langchain_openai import OpenAIEmbeddings # OpenAIEmbeddings is a vector store that uses OpenAI's GPT-3 to generate embeddings

# Define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__)) # abspath returns the absolute path of the file | __file__ is the path of the current file
file_path = os.path.join(current_dir, 'books', 'odyssey.txt') # join combines the current directory with the path to the text file
persistent_directory = os.path.join(current_dir, "db", "chroma_db") # join combines the current directory with the path to the database directory

# check if the Chroma vector store already exists
if not os.path.exists(persistent_directory): # if the directory does not exist
    print("Persistent directory does not exist. Initializing vector store...")  

    # Ensure that the text file exists
    if not os.path.exists(file_path): # if the text file does not exist
        raise FileNotFoundError(f"Text file not found at {file_path}") # raise an error

    # Read the text content from the file
    loader = TextLoader(file_path, encoding="utf-8") # create a TextLoader object with the file path and encoding
    documents = loader.load() # load the text content from the file

    # Split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0) # create a CharacterTextSplitter object with chunk size and overlap
    docs = text_splitter.split_documents(documents) # split the documents into chunks

    # Display information about the split documents
    print("--------- Document chunks Information ---------")
    print(f"Number of documents chunks: {len(docs)}") # print the number of document chunks
    print(f"Sampel document chunk: {docs[0]}") # print a sample document chunk

    # Create embeddings using OpenAI
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    ) # create an OpenAIEmbeddings object with the model name

    print("Finished creating embeddings.")

    # Create a vector store and persist it automatically
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persistent_directory
    ) # create a Chroma object with the document chunks, embeddings, and persistent directory



    print("Finished creating vector store.")

else: # if the directory exists
    # Load the existing vector store
    print("Persistent directory exists. Loading vector store...")
    db = Chroma(persist_directory=persistent_directory)