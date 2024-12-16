import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader

 

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "odyssey.txt")
persistent_directory = os.path.join(current_dir, "db", "croma_db")  #defining the directory having text file and persistent directory

if not os.path.exists(persistent_directory):   #check if chroma vector store already exists
    print("Persistent directory doesn't exist. Initializing vector store...")

    if not os.path.exists(file_path):   #check if the text file exists.
        raise FileNotFoundError(
            f"The file {file_path} doesn't exist. Please check the path."
        )
    
    loader = TextLoader(file_path)
    documents = loader.load()     #loading content from file

    text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
    docs = text_splitter.split_documents(documents)    #splits document into chunks

    print("\n ---- Document Chunks Information ----")    #information of the split documents
    print(f"Number of chunks: {len(docs)}")
    print(f"Sample Chunk: \n {docs[0].page_content} \n")

    print("\n ---- Creating Embeddings ----")
    embeddings = OpenAIEmbeddings(
        model = "text-embedding-ada-002",
        api_key=""
    )

    print("\n ---- Creating Vector Store ----")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory = persistent_directory
    )

else:
    print("Vector store already exixts. No need to initialize.")

    