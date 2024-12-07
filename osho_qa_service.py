# Suppress warnings - must be before any imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
import logging
# Suppress all warnings
warnings.filterwarnings('ignore')
# Specific suppressions
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*benefit from vacuuming.*')
warnings.filterwarnings('ignore', message='.*sparse_softmax_cross_entropy.*')

# Suppress all logging
logging.getLogger().setLevel(logging.ERROR)
# Suppress TensorFlow logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
from typing import Dict, List
import chromadb
from chromadb.utils import embedding_functions

def clean_text(text: str) -> str:
    """Clean the text by removing extra spaces and formatting."""
    # Remove multiple spaces
    text = ' '.join(text.split())
    # Remove unnecessary line breaks
    text = text.replace('\n', ' ')
    
    # Remove text before first complete sentence
    if '.' in text:
        # Split by period and remove any incomplete sentence at start
        sentences = text.split('.')
        # Remove first part if it seems like a partial sentence
        if len(sentences) > 1:  # Only if there are multiple sentences
            sentences = sentences[1:]  # Remove first part
        text = '.'.join(sentences)
        text = text.strip()  # Remove leading/trailing whitespace
        if text:  # Add period back if text is not empty
            text += '.'
    
    return text

def get_answer_from_osho(question: str, n_results: int = 5) -> Dict:
    """
    Get answer from Osho's books based on the question.
    
    Args:
        question (str): The question to ask
        n_results (int): Number of relevant passages to return
        
    Returns:
        Dict: A dictionary containing the question and formatted answer with sources
    """
    # Initialize ChromaDB client
    db_dir = os.path.join(os.getcwd(), "vector_db")
    if not os.path.exists(db_dir):
        # If local path doesn't exist, download from Hugging Face
        from huggingface_hub import snapshot_download
        db_dir = snapshot_download(repo_id="harithapliyal/osho-vector-db")
    client = chromadb.PersistentClient(path=db_dir)
    
    # Initialize embedding function
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Get the collection
    collection = client.get_collection(
        name="osho_books",
        embedding_function=embedding_function
    )
    
    # Query the collection
    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )
    
    # Format the answer
    answer_parts = []
    for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        answer_part = {
            "passage_number": i + 1,
            "book": metadata['book'],
            "text": clean_text(doc.strip())
        }
        answer_parts.append(answer_part)
    
    # Create the response
    response = {
        "question": question,
        "answer_passages": answer_parts,
        "total_passages": len(answer_parts)
    }
    
    return response

def save_qa_to_file(qa_response: Dict, output_file: str = None):
    """
    Save the Q&A response to a JSON file.
    
    Args:
        qa_response (Dict): The Q&A response to save
        output_file (str): Optional output file path. If None, generates a filename
    """
    if output_file is None:
        # Create answers directory if it doesn't exist
        answers_dir = os.path.join(os.getcwd(), "answers")
        os.makedirs(answers_dir, exist_ok=True)
        
        # Generate filename from question
        filename = f"answer_{qa_response['question'][:30].lower().replace(' ', '_')}.json"
        output_file = os.path.join(answers_dir, filename)
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(qa_response, f, ensure_ascii=False, indent=2)
    
    return output_file

if __name__ == "__main__":
    # Example usage
    question = "What is the nature of consciousness?"
    
    # Get answer
    response = get_answer_from_osho(question)
    
    # Save to file
    output_file = save_qa_to_file(response)
    
    # Print the response
    print(f"\nQuestion: {response['question']}\n")
    for passage in response['answer_passages']:
        print(f"\nPassage {passage['passage_number']}:")
        print(f"Book: {passage['book']}")
        print(f"Text: {passage['text'][:200]}...")
        print("-" * 80)
    
    print(f"\nResponse saved to: {output_file}")
