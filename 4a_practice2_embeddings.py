from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document  

# Practice -2 Using Hugging Face Sentence Transformers for Text Embeddings

# Example documents as Document objects
documents = [
    Document(page_content="Artificial intelligence is transforming the world."),
    Document(page_content="Machine learning is a subset of AI focusing on data-driven algorithms."),
]

# Split documents
text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=10)
docs = text_splitter.split_documents(documents)

# Hugging Face embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(docs, embeddings)

# Query the vector store
query = "What does AI focus on?"
retriever = vector_store.as_retriever(search_type="similarity")
results = retriever.invoke(query)

# Display results
print("Query Results:")
for result in results:
    print(result.page_content)
