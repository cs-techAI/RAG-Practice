import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from creds import API_KEY


# Practice -1  Basic Vector Store Creation Using OpenAI Embeddings


current_dir = os.path.dirname(os.path.abspath(__file__))         # Load and split documents
file_path = os.path.join(current_dir, "books", "audi.txt")
loader = TextLoader(file_path)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)


embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=API_KEY)    # Use OpenAI embeddings
vector_store = Chroma.from_documents(docs, embeddings)


query = "What is the document about?"         # Query the vector store

retriever = vector_store.as_retriever(search_type="similarity")
results = retriever.invoke(query)

print("Query Results:")
for result in results:
    print(result.page_content)
