---
title: Ask Osho
emoji: 🧘‍♂️
colorFrom: yellow
colorTo: green
sdk: streamlit
sdk_version: 1.29.0
app_file: streamlit_app.py
pinned: false
license: apache-2.0
---

# Ask Osho - AI-Powered Question Answering System

This is an Streamlit-UI - AI-powered question-answering system that allows you to ask questions about Osho's teachings. The system uses a vector database built from Osho's books to provide relevant answers along with source references.


## Features

- Ask questions about Osho's teachings
- Choose from example questions or ask your own
- Get answers with references to specific books
- Clean and intuitive user interface
- Additional book recommendations for further reading

## Technology Stack

- Streamlit for the web interface
- ChromaDB for vector storage
- Sentence Transformers for embeddings
- Hugging Face for deployment 

## How it Works

The application uses:
- Sentence transformers for text embeddings
- ChromaDB for vector similarity search
- Streamlit for the user interface

## Usage

1. Select an example question or type your own
2. Click "Please Answer Osho"
3. View the answer and book references

## Development & Local Installation.

To run this application locally:

```bash
git clone https://github.com/dasarpai/Media-Publication-TWO
cd Media-Publication-TWO
python -m venv .venv 
.venv\Script\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deployment

This application is deployed on Hugging Face Spaces.

## Links 
- https://huggingface.co/spaces/harithapliyal/ask-osho

