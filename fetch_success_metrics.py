"""
NBA Success Metrics Fetcher
Fetches team standings, scoring leaders, and playoff scores for SNA analysis.
"""

import os
import time
import pandas as pd
from nba_api.stats.endpoints import (
    LeagueStandings,
    LeagueDashPlayerStats,
    TeamYearByYearStats,
)
from nba_api.stats.static import teams

# Configuration
START_SEASON = 2014
END_SEASON = 2024
DATA_DIR = 'data'
REQUEST_DELAY = 0.6  # seconds between API calls


def generate_seasons(start_year, end_year):
    """Generate season strings like '2020-21'."""
    seasons = []
    for year in range(start_year, end_year + 1):
        next_year = str(year + 1)[-2:]
        seasons.append(f"{year}-{next_year}")
    return seasons


def fetch_team_standings(season):
    """
    Fetch team WinPCT and PlayoffRank for a season.
    Output: data/{season}/team_standings.csv
    """
    print(f"  Fetching team standings for {season}...")
    try:
        standings = LeagueStandings(
            league_id='00',
            season=season,
            season_type='Regular Season'
        )
        df = standings.get_data_frames()[0]
        
        # Select only needed columns
        result = df[['TeamID', 'TeamName', 'WinPCT', 'PlayoffRank']].copy()
        
        output_path = os.path.join(DATA_DIR, season, 'team_standings.csv')
        result.to_csv(output_path, index=False)
        print(f"    Saved {len(result)} teams to {output_path}")
        return True
    except Exception as e:
        print(f"    Error: {e}")
        return False


def fetch_scoring_leaders(season):
    """
    Fetch top 100 scoring leaders for a season.
    Output: data/{season}/scoring_leaders.csv
    """
    print(f"  Fetching scoring leaders for {season}...")
    try:
        # Use LeagueDashPlayerStats instead of LeagueLeaders (which fails for older seasons)
        stats = LeagueDashPlayerStats(
            league_id_nullable='00',
            per_mode_detailed='PerGame',
            season=season,
            season_type_all_star='Regular Season',
            measure_type_detailed_defense='Base'
        )
        df = stats.get_data_frames()[0]
        
        # Sort by PTS descending and take top 100
        result = df.sort_values(by='PTS', ascending=False).head(100).copy()
        
        output_path = os.path.join(DATA_DIR, season, 'scoring_leaders.csv')
        result.to_csv(output_path, index=False)
        print(f"    Saved {len(result)} leaders to {output_path}")
        return True
    except Exception as e:
        print(f"    Error: {e}")
        return False


def calculate_playoff_score(playoff_wins, made_playoffs, play_in_loss=False):
    """
    Calculate playoff score based on postseason performance.
    
    Scoring:
    - Missed playoffs: 0
    - Lost in play-in: 0.5
    - Qualified (1st round loss): 2 (1 for qualifying + 1 for 1st round)
    - 2nd round loss: 3
    - Conference Finals loss: 4
    - Finals loss: 5
    - Champion: 6
    
    Approximation using playoff wins:
    - 0 wins in playoffs = 1st round loss (score 2)
    - 1-3 wins = still 1st round (score 2) 
    - 4-7 wins = 2nd round (score 3)
    - 8-11 wins = conf finals (score 4)
    - 12-15 wins = finals loss (score 5)
    - 16+ wins = champion (score 6)
    """
    if not made_playoffs:
        if play_in_loss:
            return 0.5
        return 0.0
    
    # Made playoffs = at least 2 points
    if playoff_wins >= 16:
        return 6.0  # Champion
    elif playoff_wins >= 12:
        return 5.0  # Finals loss
    elif playoff_wins >= 8:
        return 4.0  # Conference Finals loss
    elif playoff_wins >= 4:
        return 3.0  # 2nd round loss
    else:
        return 2.0  # 1st round loss


def fetch_playoff_scores(season):
    """
    Fetch playoff scores for all teams in a season.
    Uses team standings + playoff stats to calculate custom score.
    Output: data/{season}/playoff_scores.csv
    """
    print(f"  Fetching playoff scores for {season}...")
    
    try:
        # Get team standings first to know playoff rank
        standings = LeagueStandings(
            league_id='00',
            season=season,
            season_type='Regular Season'
        )
        standings_df = standings.get_data_frames()[0]
        time.sleep(REQUEST_DELAY)
        
        results = []
        nba_teams = teams.get_teams()
        
        for team in nba_teams:
            team_id = team['id']
            team_name = team['full_name']
            
            # Find team in standings
            team_standing = standings_df[standings_df['TeamID'] == team_id]
            if team_standing.empty:
                continue
                
            playoff_rank = team_standing['PlayoffRank'].values[0]
            
            # Check if made playoffs (rank 1-8 in each conference = made playoffs)
            # PlayoffRank 0 or >10 typically means missed
            made_playoffs = 1 <= playoff_rank <= 10
            
            # Get playoff wins for this team this season
            playoff_wins = 0
            try:
                team_stats = TeamYearByYearStats(
                    team_id=team_id,
                    season_type_all_star='Playoffs',
                    per_mode_simple='Totals'
                )
                playoff_df = team_stats.get_data_frames()[0]
                
                # The YEAR column uses format like "2023-24" - match against our season
                if 'YEAR' in playoff_df.columns:
                    season_stats = playoff_df[playoff_df['YEAR'] == season]
                    if not season_stats.empty:
                        playoff_wins = int(season_stats['WINS'].values[0])
                    else:
                        # Season not found = no playoff games = didn't make playoffs
                        made_playoffs = False
                else:
                    made_playoffs = False
                    
                time.sleep(REQUEST_DELAY)
            except Exception as e:
                # No playoff data = didn't make playoffs
                made_playoffs = False
            
            # Play-in started in 2020-21
            season_year = int(season[:4])
            play_in_exists = season_year >= 2020
            
            # Approximate play-in loss: rank 7-10 but 0 playoff wins and made_playoffs is still True from standings
            play_in_loss = play_in_exists and (7 <= playoff_rank <= 10) and playoff_wins == 0 and not made_playoffs
            
            score = calculate_playoff_score(playoff_wins, made_playoffs or playoff_wins > 0, play_in_loss)
            
            results.append({
                'TeamID': team_id,
                'TeamName': team_name,
                'PlayoffRank': playoff_rank,
                'PlayoffWins': playoff_wins,
                'PlayoffScore': score
            })
        
        result_df = pd.DataFrame(results)
        output_path = os.path.join(DATA_DIR, season, 'playoff_scores.csv')
        result_df.to_csv(output_path, index=False)
        print(f"    Saved {len(result_df)} teams to {output_path}")
        return True
        
    except Exception as e:
        print(f"    Error: {e}")
        return False


def process_season(season):
    """Process all metrics for a single season."""
    print(f"\n{'='*50}")
    print(f"Processing season: {season}")
    print(f"{'='*50}")
    
    # Ensure directory exists
    season_dir = os.path.join(DATA_DIR, season)
    if not os.path.exists(season_dir):
        os.makedirs(season_dir)
    
    fetch_team_standings(season)
    time.sleep(REQUEST_DELAY)
    
    fetch_scoring_leaders(season)
    time.sleep(REQUEST_DELAY)
    
    fetch_playoff_scores(season)


def main():
    seasons = generate_seasons(START_SEASON, END_SEASON)
    print(f"Fetching success metrics for {len(seasons)} seasons: {seasons}")
    
    for season in seasons:
        process_season(season)
    
    print(f"\n{'='*50}")
    print("All seasons complete!")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
