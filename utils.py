import os
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

def process_documents(pdfs):
    """
    Process the Tax Regime PDF document: load, split, embed, and store persistently.
    Returns a persistent Chroma vector store.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_paths = []
        for pdf in pdfs:
            path = os.path.join(temp_dir, pdf.name)
            with open(path, "wb") as f:
                f.write(pdf.getbuffer())
            pdf_paths.append(path)
        
        # Load documents from the PDF(s)
        documents = []
        for path in pdf_paths:
            loader = PDFPlumberLoader(path)
            documents.extend(loader.load())
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,  
            chunk_overlap=150  
        )
        splits = text_splitter.split_documents(documents)
        
        # Instantiate the embeddings model
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        # Create embeddings and build a persistent vector store in './chroma_db'
        vector_store = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        vector_store.persist()  # Save data permanently
        
        return vector_store

def load_vector_store():
    """
    Load the existing persistent Chroma vector store.
    """
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma(
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )

def get_retriever():
    """
    Initialize and return the vector store retriever.
    """
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    try:
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory="./chroma_db"
        )
        return vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})
    except Exception as e:
        print(f"Error initializing vector store: {e}")
        return None
