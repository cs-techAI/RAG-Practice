import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
books_dir = os.path.join(current_dir,"books")
persistent_dir = os.path.join(db_dir, "chroma_db_withMetaData")

print(f"Book directory: {books_dir}")
print(f"Persistent directory: {persistent_dir}")

if not os.path.exists(persistent_dir):
    print("Persistant directory doesn't exixt. Initializing vector store....")

    if not os.path.exists(books_dir):
        raise FileNotFoundError("File doesn't exist.")
    
    books_file = [f for f in os.listdir(books_dir) if f.endswith(".txt")]

    documents = []
    for book_file in books_file:
        file_path = os.path.join(books_dir, book_file)
        loader = TextLoader(file_path)
        book_docs = loader.load()
        for doc in book_docs:
            doc.metadata = {"source": book_file}
            documents.append(doc)

    text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
    docs = text_splitter.split_documents(documents)

    print("------- Document Chunk Information --------")
    print(f"Length of document chunk is: {len(docs)}")

    print("-------- Creating Embeddings ---------")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        api_key=os.getenv("OPENAI_API_KEY")
        )


    print("-----------Creating and Persisting Vector Store------------")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_dir
    )                          # persistence is useful for large-scale applications where you don't 
                    # want to recreate the embeddings or reload documents every time the application starts.


    print("---------- Finished Creating and Persiting Vector Store ----------")

else:
    print("Vector store alrady exists. No need to create.")
