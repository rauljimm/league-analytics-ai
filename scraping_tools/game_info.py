import sys
import os
import time
import requests
import pandas as pd
from typing import Dict, Any, List, Set

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.league_scraper import LeagueScraper

def load_existing_matches(file_path: str) -> Set[str]:
    """Load match IDs from an existing CSV to prevent duplicate processing."""
    existing_matches = set()
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            if 'match_id' in df.columns:
                existing_matches = set(df['match_id'].astype(str))
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")
    return existing_matches

def load_match_ids(file_path: str) -> List[str]:
    """Load match IDs from a text file."""
    match_ids = []
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                match_ids = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")
    else:
        print(f"File not found: {file_path}")
    return match_ids

def get_region_url(scraper: LeagueScraper, match_id: str) -> str:
    """Determine the appropriate region URL based on the match ID prefix."""
    prefix = match_id.split('_')[0]
    if prefix == 'EUW1':
        return scraper.riot_eu_url
    elif prefix == 'NA1':
        return scraper.riot_na_url
    elif prefix == 'KR':
        return scraper.riot_kr_url
    else:
        raise ValueError(f"Unsupported region for match ID {match_id}: {prefix}")

def process_match(scraper: LeagueScraper, match_id: str) -> Dict[str, Any]:
    """Extract features from a single match using match and timeline data."""
    # Determine the region for API requests
    try:
        region_url = get_region_url(scraper, match_id)
    except ValueError as e:
        print(f"[{match_id}] {e}")
        return None

    # Define API endpoints for match and timeline data
    match_endpoint = f"lol/match/v5/matches/{match_id}"
    timeline_endpoint = f"lol/match/v5/matches/{match_id}/timeline"

    # Fetch match and timeline data
    try:
        match_data = scraper.make_request(region_url, match_endpoint)
        time.sleep(1.2)  # Delay to respect API rate limits
        timeline_data = scraper.make_request(region_url, timeline_endpoint)
        time.sleep(1.2)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            print(f"[{match_id}] Rate limit exceeded. Waiting for 3 seconds...")
            time.sleep(3)
            return None
        elif e.response.status_code == 403:
            print(f"[{match_id}] Access forbidden: {e}")
            return None
        else:
            print(f"[{match_id}] HTTP error: {e}")
            return None
    except Exception as e:
        print(f"[{match_id}] Failed to fetch data: {e}")
        return None

    if not match_data or not timeline_data:
        print(f"[{match_id}] Invalid or missing data")
        return None

    # Initialize match features dictionary
    match_features = {
        "match_id": match_id,
        "game_duration": 0,
        "blue_win": False,
        "blue_towers": 0, "red_towers": 0,
        "blue_dragons": 0, "red_dragons": 0,
        "blue_barons": 0, "red_barons": 0,
        "blue_inhibitors": 0, "red_inhibitors": 0,
        "blue_heralds": 0, "red_heralds": 0,
        "blue_gold": 0, "red_gold": 0,
        "blue_kills": 0, "red_kills": 0,
        "blue_deaths": 0, "red_deaths": 0,
        "blue_assists": 0, "red_assists": 0,
        "blue_cs": 0, "red_cs": 0,
        "blue_vision_score": 0, "red_vision_score": 0,
        "blue_champions": [], "red_champions": []
    }

    # Extract game duration and filter out short games (less than 15 minutes)
    match_features["game_duration"] = match_data["info"]["gameDuration"]
    if match_features["game_duration"] < 900:
        print(f"[{match_id}] Game too short or remake (duration: {match_features['game_duration']} seconds)")
        return None

    # Extract team-level stats
    for team in match_data["info"]["teams"]:
        team_id = team["teamId"]
        prefix = "blue" if team_id == 100 else "red"
        
        if prefix == "blue":
            match_features["blue_win"] = team["win"]
        
        objectives = team["objectives"]
        match_features[f"{prefix}_towers"] = objectives["tower"]["kills"]
        match_features[f"{prefix}_dragons"] = objectives["dragon"]["kills"]
        match_features[f"{prefix}_barons"] = objectives["baron"]["kills"]
        match_features[f"{prefix}_inhibitors"] = objectives["inhibitor"]["kills"]
        match_features[f"{prefix}_heralds"] = objectives["riftHerald"]["kills"]

    # Extract player-level stats
    for participant in match_data["info"]["participants"]:
        team_id = participant["teamId"]
        prefix = "blue" if team_id == 100 else "red"
        
        match_features[f"{prefix}_kills"] += participant["kills"]
        match_features[f"{prefix}_deaths"] += participant["deaths"]
        match_features[f"{prefix}_assists"] += participant["assists"]
        match_features[f"{prefix}_gold"] += participant["goldEarned"]
        match_features[f"{prefix}_cs"] += participant["totalMinionsKilled"] + participant["neutralMinionsKilled"]
        match_features[f"{prefix}_vision_score"] += participant["visionScore"]
        match_features[f"{prefix}_champions"].append(participant["championName"])

    # Compute derived features
    match_features["gold_diff"] = match_features["blue_gold"] - match_features["red_gold"]
    match_features["tower_diff"] = match_features["blue_towers"] - match_features["red_towers"]
    match_features["dragon_diff"] = match_features["blue_dragons"] - match_features["red_dragons"]
    match_features["blue_gold_per_min"] = match_features["blue_gold"] / (match_features["game_duration"] / 60)
    match_features["red_gold_per_min"] = match_features["red_gold"] / (match_features["game_duration"] / 60)

    # Process timeline data for specific time points
    target_minutes = [5, 10, 15, 20, 25]
    timeline_features = {}
    for minute in target_minutes:
        timeline_features.update({
            f"min_{minute}_gold_diff": 0,
            f"min_{minute}_kill_diff": 0,
            f"min_{minute}_blue_towers": 0,
            f"min_{minute}_red_towers": 0,
            f"min_{minute}_blue_dragons": 0,
            f"min_{minute}_red_dragons": 0,
            f"min_{minute}_blue_wards": 0,
            f"min_{minute}_red_wards": 0
        })

    # Track events over time
    blue_kills = []
    red_kills = []
    tower_events = []
    dragon_events = []
    ward_events = []

    # Extract events from timeline
    for frame in timeline_data["info"]["frames"]:
        for event in frame["events"]:
            timestamp = event["timestamp"] / 1000 / 60  # Convert timestamp to minutes
            if timestamp <= max(target_minutes):
                if event["type"] == "CHAMPION_KILL":
                    killer_team = 100 if event["killerId"] <= 5 else 200
                    if killer_team == 100:
                        blue_kills.append(timestamp)
                    else:
                        red_kills.append(timestamp)
                elif event["type"] == "BUILDING_KILL" and event["buildingType"] == "TOWER_BUILDING":
                    tower_events.append({"timestamp": timestamp, "teamId": event["teamId"]})
                elif event["type"] == "ELITE_MONSTER_KILL" and event["monsterType"] == "DRAGON":
                    killer_team = 100 if event["killerId"] <= 5 else 200
                    dragon_events.append({"timestamp": timestamp, "teamId": killer_team})
                elif event["type"] == "WARD_PLACED":
                    creator_team = 100 if event["creatorId"] <= 5 else 200
                    ward_events.append({"timestamp": timestamp, "teamId": creator_team})

    # Extract gold differences from timeline frames
    last_gold_diff = {}
    for frame in timeline_data["info"]["frames"]:
        timestamp = frame["timestamp"] / 1000 / 60
        blue_gold = 0
        red_gold = 0
        for pid, pf in frame["participantFrames"].items():
            team_id = 100 if int(pid) <= 5 else 200
            if team_id == 100:
                blue_gold += pf["totalGold"]
            else:
                red_gold += pf["totalGold"]
        
        for minute in target_minutes:
            if minute - 0.5 <= timestamp <= minute + 0.5:
                timeline_features[f"min_{minute}_gold_diff"] = blue_gold - red_gold
        last_gold_diff[timestamp] = blue_gold - red_gold

    # Fill in missing gold differences with the last known value
    for minute in target_minutes:
        if timeline_features[f"min_{minute}_gold_diff"] == 0:
            last_timestamp = max([t for t in last_gold_diff.keys() if t <= minute], default=0)
            if last_timestamp > 0:
                timeline_features[f"min_{minute}_gold_diff"] = last_gold_diff[last_timestamp]

    # Compute event counts at each target minute
    for minute in target_minutes:
        timeline_features[f"min_{minute}_kill_diff"] = (
            sum(1 for t in blue_kills if t <= minute) -
            sum(1 for t in red_kills if t <= minute)
        )
        timeline_features[f"min_{minute}_blue_towers"] = sum(
            1 for e in tower_events if e["timestamp"] <= minute and e["teamId"] == 100
        )
        timeline_features[f"min_{minute}_red_towers"] = sum(
            1 for e in tower_events if e["timestamp"] <= minute and e["teamId"] == 200
        )
        timeline_features[f"min_{minute}_blue_dragons"] = sum(
            1 for e in dragon_events if e["timestamp"] <= minute and e["teamId"] == 100
        )
        timeline_features[f"min_{minute}_red_dragons"] = sum(
            1 for e in dragon_events if e["timestamp"] <= minute and e["teamId"] == 200
        )
        timeline_features[f"min_{minute}_blue_wards"] = sum(
            1 for e in ward_events if e["timestamp"] <= minute and e["teamId"] == 100
        )
        timeline_features[f"min_{minute}_red_wards"] = sum(
            1 for e in ward_events if e["timestamp"] <= minute and e["teamId"] == 200
        )

    # Combine match and timeline features
    return {**match_features, **timeline_features}

def save_to_csv(features_list: List[Dict[str, Any]], csv_path: str):
    """Append extracted features to a CSV file, avoiding duplicates."""
    if not features_list:
        print("No data to save")
        return
    
    df = pd.DataFrame(features_list)
    
    try:
        if os.path.exists(csv_path):
            existing_df = pd.read_csv(csv_path)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=["match_id"])
            combined_df.to_csv(csv_path, index=False)
            print(f"Appended {len(df)} new matches to {csv_path}")
        else:
            df.to_csv(csv_path, index=False)
            print(f"Created CSV at {csv_path} with {len(df)} matches")
    except Exception as e:
        print(f"Failed to save to {csv_path}: {e}")

def main():
    """Main function to scrape match data and save features to CSV."""
    # Initialize the scraper
    scraper = LeagueScraper()
    matches_file = 'data/matches.txt'
    csv_path = 'data/match_features.csv'

    # Load existing match IDs to avoid duplicates
    seen_matches = load_existing_matches(csv_path)
    print(f"Loaded {len(seen_matches)} existing match IDs from {csv_path}")

    # Load match IDs from text file
    match_ids = load_match_ids(matches_file)
    print(f"Found {len(match_ids)} match IDs in {matches_file}")

    # Filter out already processed match IDs
    match_ids = [mid for mid in match_ids if mid not in seen_matches]
    print(f"Processing {len(match_ids)} new match IDs")

    if not match_ids:
        print("No new matches to process")
        return

    # Process matches in batches
    all_features = []
    count = 0

    for match_id in match_ids:
        count += 1
        print(f"Processing match {count}/{len(match_ids)}: {match_id}")
        features = process_match(scraper, match_id)
        if features:
            all_features.append(features)
        
        # Save every 10 matches or at the end
        if len(all_features) >= 10 or count == len(match_ids):
            save_to_csv(all_features, csv_path)
            all_features = []
        
        time.sleep(1.2)  # Delay to respect API rate limits

    print(f"Finished processing {count} matches")

if __name__ == '__main__':
    main()