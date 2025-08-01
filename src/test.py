import requests

url = "https://policy-api-881938266006.asia-south1.run.app/hackrx/run"
headers = {
    "Authorization": "Bearer 1f6a94eede58fbcb41c3980661fcab4ce4359fcac8b58038ba206d000dda74a2",  # replace with actual key or dummy for local
    "Content-Type": "application/json"
}
data = {
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
}

try:
    response = requests.post(url, headers=headers, json=data)
    print("Status Code:", response.status_code)
    print("Response:")
    print(response.json())
except Exception as e:
    print("❌ Request failed:", str(e))
response = requests.post(url, headers=headers, json=data)
print(response.status_code)
print(response.json())
