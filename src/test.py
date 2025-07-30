import requests

url = "http://localhost:5050/hackrx/run"
headers = {
    "Authorization": "Bearer 1f6a94eede58fbcb41c3980661fcab4ce4359fcac8b58038ba206d000dda74a2",  # replace with actual key or dummy for local
    "Content-Type": "application/json"
}
data = {
    "documents": "https://example.com/policy.pdf",  # optional field, will be ignored
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

response = requests.post(url, headers=headers, json=data)
print(response.status_code)
print(response.json())
