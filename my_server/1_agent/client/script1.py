import requests

response = requests.post(
    "http://localhost:8000/agent-1/invoke",
    json={"input": "I am in London, what should I wear?"},
)

print(response.json())
