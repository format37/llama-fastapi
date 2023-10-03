import requests
import argparse

def call_generate_api(model_filename, start_text):
    # URL of the FastAPI server
    url = "http://localhost:8091/generate/"
    
    """MASTER_CONFIG = {
        'context_window': 16,
        'd_model': 128,
        'epochs': 16000,
        'log_interval': 100,
        'batch_size': 12288,
        'n_layers': 4,
        'n_heads': 8,
    }"""
    MASTER_CONFIG = {
        'context_window': 16,
        'd_model': 768,
        'epochs': 1000,
        'log_interval': 10,
        'batch_size': 2,
        'n_layers': 12,
        'n_heads': 12,
    }

    params = {
        "start_text": start_text, 
        "max_new_tokens": 1000,
        "model_filename": model_filename,
        "dataset_filepath": "data/tiny_stories/TinyStoriesV2-GPT4-train.txt",
        "MASTER_CONFIG": MASTER_CONFIG
        }

    
    # Make the API call
    response = requests.get(url, json=params)
    
    if response.status_code == 200:
        return response.json()["generated_text"]
    else:
        return f"Error: {response.status_code}"


def main():
    parser = argparse.ArgumentParser(description='Process some command-line parameters.')
    # Add arguments
    parser.add_argument('--model_filename', type=str, default='llama.pt', help='The name of the model file.')
    parser.add_argument('--start_text', type=str, default='This', help='The start text for processing.')
    
    # Parse arguments
    args = parser.parse_args()

    # Access arguments
    model_filename = args.model_filename
    start_text = args.start_text

    print(f"Model Filename: {model_filename}")
    print(f"Start Text: {start_text}")

    
    generated_text = call_generate_api(model_filename, start_text)
    print(f"Generated text: {generated_text}")


if __name__ == "__main__":
    main()
