import requests

BASE_URL = "http://127.0.0.1:8000"

def test_workflow():
    # 1. Test Ingestion
    print("Testing document ingestion...")
    sample_doc = {
        "title": "Tokyo Travel Guide Sample",
        "content": "Senso-ji is Tokyo's oldest and most significant Buddhist temple, located in Asakusa. \n\nTsukiji Outer Market is famous for fresh seafood, street food, and sushi breakfast stalls.",
        "source_type": "text"
    }
    
    ingest_res = requests.post(f"{BASE_URL}/ingest", json=sample_doc)
    print("Ingest Response:", ingest_res.json())
    
    if ingest_res.status_code == 200:
        # 2. Test Chat / Search Retrieval
        print("\nTesting AI Chat based on documents...")
        chat_query = {"question": "What can I eat at Tsukiji Market?"}
        chat_res = requests.post(f"{BASE_URL}/chat", json=chat_query)
        print("Chat Response:", chat_res.json())

if __name__ == "__main__":
    test_workflow()