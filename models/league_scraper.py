import requests
from config import API_KEY

class LeagueScraper:
    def __init__(self):       
        self.api_key = API_KEY
        self.euw1_url = "https://euw1.api.riotgames.com/"
        self.na1_url = "https://na1.api.riotgames.com/"
        self.kr_url = "https://kr.api.riotgames.com/"
        self.riot_eu_url = "https://europe.api.riotgames.com/"
        self.riot_na_url = "https://americas.api.riotgames.com/"
        self.riot_kr_url = "https://asia.api.riotgames.com/"
    def make_request(self, region_url, endpoint, params=None):
        headers = {
            "X-Riot-Token": self.api_key,
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(f"{region_url}{endpoint}", headers=headers, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")