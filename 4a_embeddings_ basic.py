import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from creds import API_KEY


current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "odyssey.txt")
db_dir = os.path.join(current_dir, "db")

if not os.path.exists(file_path):
    raise FileNotFoundError(f"File {file_path} doesn't exist.")

loader = TextLoader(file_path)
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap= 0)
docs = text_splitter.split_documents(documents)


print("\n---- Document Chunk Information ----")
print(f"\n Number of document chunks: {len(docs)}")
print(f"Sample Chunk: \n{docs [0].page_content}\n")


def create_vector_store(docs, embeddings, store_name):
    persistent_dir = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_dir):
        print(f"\n ---- Creating Vector Store {store_name}")
        Chroma.from_documents(docs, embeddings, persist_directory=persistent_dir)
        print(f"\n ---- Finished Creating Vector Store {store_name} ----\n")

    else:
        print(f" ---- Vector Store Already Exists. No need to initialize ----")




print("\n ----- Using OpenAI Embeddings -----")
openai_embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key= API_KEY)
create_vector_store(docs, openai_embeddings, "chroma_db_openai")



print("\n ----- Using Hugging Face Transformers -----")
huggingface_embeddings = HuggingFaceEmbeddings(model = "sentence-transformers/all-mpnet-base-v2")
create_vector_store(docs, huggingface_embeddings, "chroma_db_huggingface")

print("Demonstration for embeddings using OpenAI and HuggingFace Completed.")


def query_vector_store(store_name, query, embedding_function):
    persistent_dir = os.path.join(db_dir, store_name)
    if os.path.exists(persistent_dir):
        print(f"\n ----- Quering the vector store {store_name} -----")
        db = Chroma(persist_directory= persistent_dir, embedding_function= embedding_function)
        retriever = db.as_retriever(search_type = "similarity-score-threshold", 
                                    search_kwags = {"k": 1, "score_threshold": 0.1})
        relevant_docs = retriever.invoke(query)


        print(f"\n ----- Relevant documents for {store_name} ------")
        for i, doc in enumerate(relevant_docs,1):
            print(f"Document {i}: \n {doc.page_content} \n")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'unknown')}\n")

    else:
        print(f"Vector store {store_name} does not exist !!!")



query = "Who is Odyssey's wife?"

query_vector_store("chroma_db_huggingface", query, huggingface_embeddings)

