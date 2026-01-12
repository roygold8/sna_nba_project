from nba_api.stats.endpoints import LeagueLeaders, LeagueDashPlayerStats
import json

season = '2014-15'
print(f"Testing for {season}...")

print("\n--- Testing LeagueLeaders (Existing) ---")
try:
    leaders = LeagueLeaders(
        league_id='00',
        per_mode48='PerGame',
        scope='S',
        season=season,
        season_type_all_star='Regular Season',
        stat_category_abbreviation='PTS'
    )
    df = leaders.get_data_frames()[0]
    print("LeagueLeaders Success!")
    print(df.head())
except Exception as e:
    print(f"LeagueLeaders Failed: {e}")

print("\n--- Testing LeagueDashPlayerStats (Proposed) ---")
try:
    # MeasureType='Base', PerMode='PerGame', PlusMinus='N', PaceAdjust='N', Rank='N', LeagueID='00', Season=season, SeasonType='Regular Season'
    # We want to sort by PTS.
    stats = LeagueDashPlayerStats(
        league_id_nullable='00',
        per_mode_detailed='PerGame',
        season=season,
        season_type_all_star='Regular Season',
        measure_type_detailed_defense='Base'
    )
    df = stats.get_data_frames()[0]
    
    # Sort by PTS descending
    df_sorted = df.sort_values(by='PTS', ascending=False).head(100)
    
    print("LeagueDashPlayerStats Success!")
    print(df_sorted[['PLAYER_ID', 'PLAYER_NAME', 'PTS']].head())
except Exception as e:
    print(f"LeagueDashPlayerStats Failed: {e}")
