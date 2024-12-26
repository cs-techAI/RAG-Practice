from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Practice - 3    Advanced: Semantic Search Across Multiple Documents


documents = [                   # Example documents
    Document (page_content= "The sun rises in the east."),
    Document (page_content= "Python is a versatile programming language."),
    Document (page_content= "The capital of France is Paris.")
]


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")   # Generate embeddings
vector_store = FAISS.from_documents(documents, embeddings)


query = "Where does the sun rise?"                                  # Semantic search query
retriever = vector_store.as_retriever(search_type="similarity")
results = retriever.invoke(query)

print("Query Results:")
for result in results:
    print(result.page_content)
