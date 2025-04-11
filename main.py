from models.summoner import Summoner
from models.league_analytics import LeagueAnalytics
from services.league_analytics_service import LeagueAnalyticsService

if __name__ == "__main__":
    # Inicializar la API y el servicio
    lol = LeagueAnalytics("Your-API-KEY", timeout=10)
    lol_service = LeagueAnalyticsService(lol)

    # Crear el Summoner
    summoner = Summoner("BzXBjorxaNfqs6GJO9kk_758-FxtJyPBlJLsrlCmnLnSdsJxbLZ5KKIW4mkktB4qtOLpxgEe79KMwQ", "tirko", "EUW")

    # Obtener estadísticas del Summoner
    stats = lol_service.get_stats_by_summoner(summoner)
    print(stats)

    print(lol_service.get_on_going_match_by_summoner(summoner))

    print(lol_service.get_stats_by_summoner(summoner))

    # Obtener las últimas 20 partidas (puedes ajustar 'count' o agregar 'queue')
    matches = lol_service.get_matches_by_summoner(summoner, count=1, queue=420)  # 420 = Ranked Solo/Duo
    if matches:
        print(f"\nSe encontraron {len(matches)} partidas:")
        for match in matches:
            #print(lol_service.get_match_details(match))
            print(f"Partida: {match['metadata']['matchId']}")
    else:
        print("No se pudieron obtener las partidas.")