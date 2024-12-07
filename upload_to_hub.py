from huggingface_hub import HfApi, create_repo
import os
import shutil

def upload_to_hub():
    # Initialize the Hugging Face API
    api = HfApi()
    
    # Your Hugging Face info
    repo_id = "harithapliyal/osho-vector-db"
    
    # Create the dataset repository if it doesn't exist
    try:
        print("Creating dataset repository...")
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=False
        )
        print("Dataset repository created successfully!")
    except Exception as e:
        print(f"Note: {str(e)}")
    
    # Local vector database path
    local_db_path = os.path.join(os.getcwd(), "vector_db")
    
    print(f"Uploading vector database from {local_db_path}")
    
    try:
        # Upload the entire directory
        api.upload_folder(
            folder_path=local_db_path,
            repo_id=repo_id,
            repo_type="dataset"
        )
        print("Successfully uploaded vector database to Hugging Face!")
    except Exception as e:
        print(f"Error uploading to Hugging Face: {str(e)}")

if __name__ == "__main__":
    upload_to_hub()
