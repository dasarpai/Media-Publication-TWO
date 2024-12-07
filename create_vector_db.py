import os
from typing import List, Dict
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
import torch

class PDFVectorizer:
    def __init__(self, pdf_dir: str, db_dir: str):
        self.pdf_dir = pdf_dir
        self.db_dir = db_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        # Initialize ChromaDB with sentence-transformers embeddings
        self.client = chromadb.PersistentClient(path=db_dir)
        # Check if GPU is available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
            device=self.device
        )
        self.collection = self.client.create_collection(
            name="osho_books",
            embedding_function=self.embedding_function
        )

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return ""

    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """Process a single PDF file and return chunks with metadata."""
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return []
        
        chunks = self.text_splitter.split_text(text)
        book_name = os.path.basename(pdf_path)
        
        return [{
            "text": chunk,
            "metadata": {
                "book": book_name,
                "chunk_index": i
            }
        } for i, chunk in enumerate(chunks)]

    def create_vector_database(self):
        """Process all PDFs and create the vector database."""
        pdf_files = [f for f in os.listdir(self.pdf_dir) if f.endswith('.pdf')]
        
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            pdf_path = os.path.join(self.pdf_dir, pdf_file)
            chunks = self.process_pdf(pdf_path)
            
            if chunks:
                # Add chunks to ChromaDB
                self.collection.add(
                    documents=[chunk["text"] for chunk in chunks],
                    metadatas=[chunk["metadata"] for chunk in chunks],
                    ids=[f"{pdf_file}_{chunk['metadata']['chunk_index']}" for chunk in chunks]
                )
                print(f"Added {len(chunks)} chunks from {pdf_file}")

if __name__ == "__main__":
    # Define directories
    pdf_dir = os.path.join(os.getcwd(), "OshoBooks")
    db_dir = os.path.join(os.getcwd(), "vector_db")
    
    # Create vector database directory if it doesn't exist
    os.makedirs(db_dir, exist_ok=True)
    
    # Initialize and run the vectorizer
    vectorizer = PDFVectorizer(pdf_dir, db_dir)
    vectorizer.create_vector_database()
    print("Vector database creation completed!")
