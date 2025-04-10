class Summoner:
    def __init__(self, puuid, game_name, tagline):
        self.puuid = puuid
        self.game_name = game_name
        self.tagline = tagline
        self.league_id = None
        self.rank_solo = None
        self.wins_solo = None
        self.losses_solo = None
        self.hot_streak = None
        self.winrate_solo = None
        self.matches_ids = None


        # Validar que los datos no sean None o vacíos
        if not self.puuid or not isinstance(self.puuid, str):
            raise ValueError("El PUUID del Summoner no es válido.")
        if not self.game_name or not isinstance(self.game_name, str):
            raise ValueError("El nombre del Summoner no es válido.")
        if not self.tagline or not isinstance(self.tagline, str):
            raise ValueError("El tagline del Summoner no es válido.")