import streamlit as st
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
import logging
# Suppress all warnings
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)

from osho_qa_service import get_answer_from_osho

# Set page config
st.set_page_config(
    page_title="Ask Osho",
    page_icon="🧘‍♂️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Title and description
st.title("🧘‍♂️ Ask Osho")
st.markdown("""
This application allows you to ask questions and receive answers from Osho's wisdom.
Choose from example questions or ask your own question.
""")

# Example questions
example_questions = [
    "What is the relationship between breath and consciousness?",
    "How can meditation help in daily life?",
    "What is the difference between mind and consciousness?",
    "What is the nature of love?",
    "How can one find inner peace?"
]

# Initialize session state for question if not exists
if 'question' not in st.session_state:
    st.session_state.question = ""

# Create columns for example questions
st.subheader("Example Questions")
cols = st.columns(3)

# Function to update question
def set_question(q):
    st.session_state.question = q

# Create buttons for example questions
for i, question in enumerate(example_questions):
    col_idx = i % 3
    if cols[col_idx].button(f"Q{i+1}: {question[:30]}...", key=f"q{i}", 
                           help=question):  # Show full question on hover
        set_question(question)

# Or ask your own question
st.subheader("Or Ask Your Own Question")
question = st.text_input("Type your question here:", 
                        value=st.session_state.question,
                        key="question_input",
                        on_change=lambda: set_question(st.session_state.question_input))

# Answer button
if st.button("Please Answer Osho", type="primary", key="answer_button"):
    if question:
        with st.spinner("Seeking wisdom in Osho's teachings..."):
            try:
                # Get answer using the service
                response = get_answer_from_osho(question)
                
                # Display the answer in a nice box
                st.markdown("---")
                st.subheader("Osho's Answer")
                
                # Main answer
                main_passage = response['answer_passages'][0]
                st.info(f"**From the book**: _{main_passage['book']}_")
                st.markdown(main_passage['text'])
                
                # Other references
                other_books = [p['book'] for p in response['answer_passages'][1:]]
                if other_books:
                    st.markdown("---")
                    st.success("**You can also explore these books:**")
                    for book in other_books:
                        st.markdown(f"- {book}")
            except Exception as e:
                st.error("Sorry, I encountered an error while processing your question. Please try again.")
                st.error(f"Error details: {str(e)}")
    else:
        st.warning("Please enter a question or select an example question.")

# Add some styling
st.markdown("""
<style>
    .stButton button {
        width: 100%;
    }
    .stButton>button:first-child {
        margin-top: 20px;
    }
    .stMarkdown {
        text-align: justify;
    }
</style>
""", unsafe_allow_html=True)
