import os
from langchain_text_splitters import (CharacterTextSplitter,
                                      RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter,
                                      TokenTextSplitter, TextSplitter)
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "audi.txt")
db_dir = os.path.join(current_dir, "db")

if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"The file {file_path} doesn't exist. Please check the path.")

loader = TextLoader(file_path)
documents = loader.load()

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002", api_key=""
)

def create_vector_store(docs, store_name):
    persistent_dir = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_dir):
        print(f"---- Creating Vector Store {store_name} ----")
        db = Chroma.from_documents(
            docs, embeddings, persist_directory=persistent_dir
        )
        print(f"---- Finished Creating Vector Store {store_name} ----")
    else:
        print(f"Vector store {store_name} already exists. No need to initialize.")

# Using Character-based Splitting
print("\n--------- Using Character-based Splitting ---------\n")
char_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
char_docs = char_splitter.split_documents(documents)
create_vector_store(char_docs, "chroma_db_char")

# Using Recursive Character-based Splitting
print("\n--------- Using Recursive Character-based Splitting ---------\n")
recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
recursive_docs = recursive_splitter.split_documents(documents)
create_vector_store(recursive_docs, "chroma_db_recursive")

# Using Sentence-based Splitting
print("\n--------- Using Sentence-based Splitting ---------\n")
sent_splitter = SentenceTransformersTokenTextSplitter(chunk_size=1000)
sent_docs = sent_splitter.split_documents(documents)
create_vector_store(sent_docs, "chroma_db_sent")

# Using Token-based Splitting
print("\n--------- Using Token-based Splitting ---------\n")
token_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)
token_docs = token_splitter.split_documents(documents)
create_vector_store(token_docs, "chroma_db_token")

# Using Generic TextSplitter (default method)
print("\n--------- Using Generic TextSplitter ---------\n")
text_splitter = TextSplitter()
text_docs = text_splitter.split_documents(documents)
create_vector_store(text_docs, "chroma_db_generic")
