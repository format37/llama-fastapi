import requests
import json

def call_train_api():
    # URL of the FastAPI server
    url = "http://localhost:8091/train/"

    dataset_filepath = 'data/telegram_export/input.txt'   
    
    MASTER_CONFIG = {
        'context_window': 16,
        'd_model': 128,
        'epochs': 8000,
        'log_interval': 100,
        'batch_size': 12288,
        'n_layers': 4,
        'n_heads': 8,
    }
    # Make the API call
    response = requests.post(url, json=MASTER_CONFIG)
    
    if response.status_code == 200:
        return response.json()["loss"]
    else:
        return f"Error: {response.status_code}"

if __name__ == "__main__":
    loss = call_train_api()
    print(f"Loss: {loss}")