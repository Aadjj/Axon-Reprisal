import requests
import logging

def fetch_threat_intel(indicator, api_url, api_key):
    try:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        params = {"ip": indicator}
        response = requests.get(api_url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f"Threat Intel Retrieval Failed: {e}")
        return {"threat_score": 0, "is_malicious": False}