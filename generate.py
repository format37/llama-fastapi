import requests

def call_generate_api(start_text):
    # URL of the FastAPI server
    url = "http://localhost:8091/generate/"
    
    # Query parameters
    # params = {"start_text": start_text}
    MASTER_CONFIG = {
        'context_window': 16,
        'd_model': 128,
        'epochs': 40000,
        'log_interval': 100,
        'batch_size': 32,
        'n_layers': 4,
        'n_heads': 8,
    }

    params = {
        "start_text": start_text, 
        "model_filename": "llama_1.0713104546070098.pt",
        "dataset_filepath": "data/telegram_export/input.txt",
        "MASTER_CONFIG": MASTER_CONFIG
        }

    
    # Make the API call
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        return response.json()["generated_text"]
    else:
        return f"Error: {response.status_code}"

if __name__ == "__main__":
    start_text = "Hello"
    generated_text = call_generate_api(start_text)
    print(f"Generated text: {generated_text}")
