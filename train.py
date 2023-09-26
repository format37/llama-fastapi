import requests

def call_train_api():
    # URL of the FastAPI server
    url = "http://localhost:8091/train/"
    
    # Query parameters
    # params = {"start_text": start_text}
    
    # Make the API call
    # response = requests.get(url, params=params)
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()["loss"]
    else:
        return f"Error: {response.status_code}"

if __name__ == "__main__":
    # start_text = "Hello"
    loss = call_train_api()
    print(f"Loss: {loss}")