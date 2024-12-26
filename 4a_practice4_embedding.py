from langchain_openai import ChatOpenAI  
from langchain_community.document_loaders import TextLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
import os
from creds import API_KEY


# Practice - 4   Document Summarization and Vectorization with LangChain


current_dir = os.path.dirname(os.path.abspath(__file__))        # Load and summarize documents
file_path = os.path.join(current_dir, "books", "audi.txt")
loader = TextLoader(file_path)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

prompt_template = ChatPromptTemplate.from_template("Summarize the following text: {text}")


chat = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=API_KEY) # Initialize ChatOpenAI or any model you're using

summaries = [chat([prompt_template.format(text=doc.page_content)]) for doc in docs]


summarized_docs = [{"page_content": summary} for summary in summaries]   # Store summarized content
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(summarized_docs, embeddings)

print("Summarization and Vector Storage Completed.")
