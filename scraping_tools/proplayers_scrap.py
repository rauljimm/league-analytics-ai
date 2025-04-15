import sys
import os
import time
import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.league_scraper import LeagueScraper

def load_existing_puuids(file_path):
    """Load existing PUUIDs from file to avoid duplicates."""
    existing_puuids = set()
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                puuid = line.strip()
                if puuid:
                    existing_puuids.add(puuid)
    return existing_puuids

def main():
    # Initialize the LeagueScraper with the API key
    league_scraper = LeagueScraper()
    regions = [
        league_scraper.euw1_url,
        league_scraper.na1_url,
        league_scraper.kr_url            
    ]
    # Define the endpoint and parameters for the request
    endpoint = "lol/league/v4/challengerleagues/by-queue/RANKED_SOLO_5x5"
    puuids_file = 'data/challenger.txt'
    
    # Load existing PUUIDs to avoid duplicates
    seen_puuids = load_existing_puuids(puuids_file)
    print(f"Loaded {len(seen_puuids)} existing PUUIDs from {puuids_file}")

    count = 0
    with open(puuids_file, 'a') as f:
        for region in regions:
            try:
                # Make the request to the API
                response = league_scraper.make_request(region, endpoint)
                entries = response['entries']
                new_puuids = 0
                
                for entry in range(len(entries)):
                    player_puuid = entries[entry]['puuid']
                    if player_puuid not in seen_puuids:
                        f.write(player_puuid + '\n')
                        seen_puuids.add(player_puuid)
                        new_puuids += 1
                        print(f"{player_puuid} added to the file\n{entry + 1} out of {len(entries)} players for {region}\n")
                    else:
                        print(f"{player_puuid} skipped (duplicate)\n{entry + 1} out of {len(entries)} players for {region}\n")
                
                count += new_puuids  # Increment count by new PUUIDs added
                print(f"Finished processing region: {region} ({new_puuids} new PUUIDs added)")

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    print(f"[{region}] Error: Too many requests... Waiting for 3 seconds.")
                    time.sleep(3)              
                elif e.response.status_code == 403:
                    print(f"[{region}] Forbidden: {e}")
                    time.sleep(3)
            except Exception as e:
                print(f"Error for region {region}: {e}")

    print("Total new players added among EUW, NA, and KR: ", count)

if __name__ == '__main__':
    main()