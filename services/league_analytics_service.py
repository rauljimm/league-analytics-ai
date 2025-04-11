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

        return None

    def get_stats_by_summoner(self, summoner: Summoner):
        puuid = summoner.puuid
        url = f"{self.league_analytics.league_base}/lol/league/v4/entries/by-puuid/{puuid}"
        
        print(f"Realizando solicitud a: {url}")
        print(f"Encabezados: {self.headers}")
        
        try:
            response = requests.get(url, headers=self.headers, timeout=self.league_analytics.timeout)
            response.raise_for_status()
            data = response.json()
            print("Información obtenida con éxito.")
            summoner.rank_solo = data['tier'] + " " + data['rank'] + data['leaguePoints']
            summoner.hot_streak = data['hotstreak']
            summoner.wins_solo = data['wins']
            summoner.x
            return data
        except requests.exceptions.HTTPError as http_err:
            print(f"Error HTTP: {http_err}, Código: {response.status_code}")
            print(f"Respuesta del servidor: {response.text}")
        except requests.exceptions.RequestException as req_err:
            print(f"Error en la solicitud: {req_err}")
        return None

    def get_match_ids(self, summoner: Summoner, count=20, queue=None):
        puuid = summoner.puuid
        url = f"{self.league_analytics.riot_base}/lol/match/v5/matches/by-puuid/{puuid}/ids?start=0&count={count}"
        if queue:
            url += f"&queue={queue}"  # Ejemplo: queue=420 para Ranked Solo/Duo
        print(f"Realizando solicitud a: {url}")
        try:
            response = requests.get(url, headers=self.headers, timeout=self.league_analytics.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            print(f"Error HTTP: {http_err}, Código: {response.status_code}")
            print(f"Respuesta del servidor: {response.text}")
        except requests.exceptions.RequestException as req_err:
            print(f"Error en la solicitud: {req_err}")
        return None

    def get_match_details(self, match_id: str):
        url = f"{self.league_analytics.riot_base}/lol/match/v5/matches/{match_id}"
        print(f"Realizando solicitud a: {url}")
        try:
            response = requests.get(url, headers=self.headers, timeout=self.league_analytics.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            print(f"Error HTTP: {http_err}, Código: {response.status_code}")
            print(f"Respuesta del servidor: {response.text}")
        except requests.exceptions.RequestException as req_err:
            print(f"Error en la solicitud: {req_err}")
        return None

    def get_matches_by_summoner(self, summoner: Summoner, count=20, queue=None):
        match_ids = self.get_match_ids(summoner, count, queue)
        if not match_ids:
            print("No se pudieron obtener los IDs de las partidas.")
            return None
        
        matches = []
        for match_id in match_ids:
            match_data = self.get_match_details(match_id)
            if match_data:
                matches.append(match_data)
        return matches

    def get_on_going_match_by_summoner(self, summoner: Summoner):
        url = f"{self.league_analytics.league_base}/lol/spectator/v5/active-games/by-summoner/{summoner.puuid}"
        print(f"Realizando solicitud a: {url}")
        try:
            response = requests.get(url, headers=self.headers, timeout=self.league_analytics.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            print(f"Error HTTP: {http_err}, Código: {response.status_code}")
            print(f"Respuesta del servidor: {response.text}")
        except requests.exceptions.RequestException as req_err:
            print(f"Error en la solicitud: {req_err}")
        return None

