import requests

def call_generate_api(start_text):
    # URL of the FastAPI server
    url = "http://localhost:8091/generate/"
    
    # Query parameters
    params = {"start_text": start_text}
    
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
