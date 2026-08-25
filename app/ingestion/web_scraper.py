import requests
from bs4 import BeautifulSoup
from app.ingestion.cleaner import clean_text

def scrape_url(url: str) -> str:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove navigation, scripts, etc.
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()
            
        text = soup.get_text(separator=' ')
        return clean_text(text)
    except Exception as e:
        raise ValueError(f"Failed to scrape URL: {str(e)}")
