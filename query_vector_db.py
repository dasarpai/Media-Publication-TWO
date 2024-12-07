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

from osho_qa_service import get_answer_from_osho

if __name__ == "__main__":
    # Example query
    query = "What is the relationship between breath and consciousness?"
    
    # Get answer using the service
    response = get_answer_from_osho(query)
    
    # Display formatted answer
    print(f"\nQuery: {response['question']}\n")
    
    # Get the first result as main answer
    main_passage = response['answer_passages'][0]
    print(f"I have discussed this idea in book '{main_passage['book']}':")
    print(f"{main_passage['text']}\n")
    
    # List other books as references
    other_books = [p['book'] for p in response['answer_passages'][1:]]
    if other_books:
        print("You can refer to other books:", ", ".join(other_books))
