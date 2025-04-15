import sys
import os
import time
import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.league_scraper import LeagueScraper

def load_existing_matches(file_path):
    """Load existing match IDs from file to avoid duplicates."""
    existing_matches = set()
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                match_id = line.strip()
                if match_id:
                    existing_matches.add(match_id)
    return existing_matches

def main():
    league_scraper = LeagueScraper()
    regions = [
        league_scraper.riot_eu_url,
        league_scraper.riot_na_url,
        league_scraper.riot_kr_url            
    ]
    endpoint = "lol/match/v5/matches/by-puuid/{puuid}/ids?type=ranked&start=0&count=20"
    count = 0
    matches_file = 'data/matches.txt'
    
    # Load existing match IDs to avoid duplicates
    seen_matches = load_existing_matches(matches_file)
    print(f"Loaded {len(seen_matches)} existing match IDs from {matches_file}")

    with open('data/challenger.txt', 'r') as f:
        lineas = f.readlines()
        
    with open(matches_file, 'a') as f2:  # Append mode to accumulate data
        for line in lineas:
            puuid = line.strip()
            if not puuid:  # Skip empty lines
                continue
            print(puuid)

            if 0 <= count < 300:
                region = regions[0]
            elif 300 <= count < 600:
                region = regions[1]
            elif 600 <= count < 900:
                region = regions[2]
            print(region)
            
            try:
                print(endpoint.format(puuid=puuid))
                response = league_scraper.make_request(region, endpoint.format(puuid=puuid))
                matches = response
                new_matches = 0
                for match in matches:
                    if match not in seen_matches:
                        f2.write(match + '\n')
                        seen_matches.add(match)
                        new_matches += 1
                        print(f"{match} added to the file ({count+1}/{len(lineas)} players processed for region {region})")
                    else:
                        print(f"{match} skipped (duplicate)")
                
                count += 1
                print(f"Finished processing PUUID #{count} in region: {region} ({new_matches} new matches added)")
                time.sleep(1.2)  # Rate limit delay

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    print(f"[{region}] Error: Too many requests... Waiting for 3 seconds.")
                    time.sleep(3)
                elif e.response.status_code == 403:
                    print(f"[{region}] Forbidden: {e}")
                    time.sleep(3)
            except Exception as e:
                print(f"Error for PUUID {puuid}: {e}")
                time.sleep(3)

if __name__ == '__main__':
    main()