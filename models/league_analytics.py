class LeagueAnalytics:
    def __init__(self, api_key: str, timeout: int = 10):
        self.api_key = api_key
        self.timeout = timeout
        self.riot_base = "https://europe.api.riotgames.com"
        self.league_base = "https://euw1.api.riotgames.com"