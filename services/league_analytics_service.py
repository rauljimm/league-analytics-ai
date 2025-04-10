import requests
from models.summoner import Summoner
from models.league_analytics import LeagueAnalytics

class LeagueAnalyticsService:
    def __init__(self, league_analytics: LeagueAnalytics):
        self.league_analytics = league_analytics
        self.headers = {
            "X-Riot-Token": league_analytics.api_key.strip()  # Evitar espacios en la clave
        }

    def get_summoner(self) -> Summoner:
        
        return

    def get_stats_by_summoner(self, summoner: Summoner):
        puuid = summoner.puuid  # Sin codificar para probar
        url = f"{self.league_analytics.league_base}/lol/league/v4/entries/by-puuid/{puuid}"
        
        print(f"Realizando solicitud a: {url}")
        print(f"Encabezados: {self.headers}")
        
        try:
            response = requests.get(url, headers=self.headers, timeout=self.league_analytics.timeout)
            response.raise_for_status()
            data = response.json()
            print("Información obtenida con éxito.")
            return data
        except requests.exceptions.HTTPError as http_err:
            print(f"Error HTTP: {http_err}, Código: {response.status_code}")
            print(f"Respuesta del servidor: {response.text}")  # Mostrar el mensaje exacto de la API
        except requests.exceptions.RequestException as req_err:
            print(f"Error en la solicitud: {req_err}")
        return None

if __name__ == "__main__":
    lol = LeagueAnalytics("RGAPI-ce29ea5f-abb9-4551-9bad-2bec8a067f88", timeout=10)
    lol_service = LeagueAnalyticsService(lol)
    summoner = Summoner("GgBF-W42DMz2i2cUQMcAZoqyxs1wkzAb3GIF1SWh0SLD2KQQdK3_b0HF7Ca2zoI1P03elUiqd1LdmA", "tirko", "EUW")
    stats = lol_service.get_stats_by_summoner(summoner)
    print(stats)