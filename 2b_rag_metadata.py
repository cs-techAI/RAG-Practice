import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_dir = os.path.join(db_dir,"chroma_db_with_metadata")

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key="")

db = Chroma(persist_directory=persistent_dir, embedding_function=embeddings)

query = "Audi's year of origin?"

retriever = db.as_retriever(
    search_type = "similarity_score_threshold",
    search_kwargs = {"k": 3, "score_threshold": 0.9}
)
relevnant_doc = retriever.invoke(query)


print("-------Relevant Documents ----------")
for i, doc in enumerate(relevnant_doc,1):
    print(f"Document {1}:\n {doc.page_content}\n")
    print(f"Source: {doc.metadata["source"]}\n")