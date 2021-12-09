import json
import requests

API_URL = 'http://127.0.0.1:8000'

# Load sample client data
with open('resources/sample_client.json') as file:
    sample_client = file.read()

# Get default probability for a client
response = requests.get(url=f'{API_URL}/predict', data=sample_client)
print(response.json())

# Get SHAP values for a client
response = requests.get(url=f'{API_URL}/shap', data=sample_client)
print(response.json())
