import requests

def call_generate_api(start_text):
    # URL of the FastAPI server
    url = "http://localhost:8091/generate/"
    
    MASTER_CONFIG = {
        'context_window': 16,
        'd_model': 128,
        'epochs': 16000,
        'log_interval': 100,
        'batch_size': 12288,
        'n_layers': 4,
        'n_heads': 8,
    }

    params = {
        "start_text": start_text, 
        "max_new_tokens": 1000,
        "model_filename": "llama_0.8231694221496582.pt",
        "dataset_filepath": "data/telegram_export/input.txt",
        "MASTER_CONFIG": MASTER_CONFIG
        }

    
    # Make the API call
    response = requests.get(url, json=params)
    
    if response.status_code == 200:
        return response.json()["generated_text"]
    else:
        return f"Error: {response.status_code}"

if __name__ == "__main__":
    start_text = "What speech"
    generated_text = call_generate_api(start_text)
    print(f"Generated text: {generated_text}")
