"""
Generate sample data for CI tests.

Creates a minimal dataset of ~32 games (4 weeks x ~8 games/week) with realistic
but synthetic data. This allows end-to-end pipeline testing without external dependencies.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Sample teams (use a subset of 8 teams for simplicity)
SAMPLE_TEAMS = ["KC", "BUF", "PHI", "SF", "DAL", "MIA", "DET", "BAL"]

# Sample season and weeks
SAMPLE_SEASON = 2022
SAMPLE_WEEKS = [1, 2, 3, 4, 5, 6]  # 6 weeks, enough for rolling windows


def generate_matchups(teams: list, week: int, season: int) -> list:
    """Generate round-robin matchups for a week."""
    np.random.seed(season * 100 + week)  # Deterministic shuffling
    shuffled = teams.copy()
    np.random.shuffle(shuffled)

    matchups = []
    for i in range(0, len(shuffled), 2):
        if i + 1 < len(shuffled):
            matchups.append((shuffled[i], shuffled[i + 1]))
    return matchups


def generate_games_df() -> pd.DataFrame:
    """Generate sample games dataframe."""
    games = []
    base_date = datetime(SAMPLE_SEASON, 9, 11)  # Week 1 start

    for week in SAMPLE_WEEKS:
        matchups = generate_matchups(SAMPLE_TEAMS, week, SAMPLE_SEASON)
        game_date = base_date + timedelta(weeks=week - 1)

        for away_team, home_team in matchups:
            # Generate realistic-ish scores
            np.random.seed(hash(f"{SAMPLE_SEASON}_{week}_{away_team}_{home_team}") % 2**31)
            home_score = np.random.randint(10, 35)
            away_score = np.random.randint(10, 35)

            game_id = f"nfl_{SAMPLE_SEASON}_{week:02d}_{away_team}_{home_team}"

            games.append({
                "game_id": game_id,
                "season": SAMPLE_SEASON,
                "week": week,
                "date": game_date,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
            })

    return pd.DataFrame(games)


def generate_markets_df(games_df: pd.DataFrame) -> pd.DataFrame:
    """Generate sample markets dataframe with odds data."""
    markets = []

    for _, game in games_df.iterrows():
        # Generate realistic spread and total
        np.random.seed(hash(game["game_id"]) % 2**31)

        # Spread based on actual score difference (with noise)
        actual_diff = game["home_score"] - game["away_score"]
        spread = -(actual_diff + np.random.uniform(-7, 7))  # Home spread
        spread = round(spread * 2) / 2  # Round to 0.5

        # Total based on actual total (with noise)
        actual_total = game["home_score"] + game["away_score"]
        total = actual_total + np.random.uniform(-5, 5)
        total = round(total * 2) / 2  # Round to 0.5

        markets.append({
            "game_id": game["game_id"],
            "season": game["season"],
            "week": game["week"],
            "close_spread": spread,
            "close_total": max(35, min(55, total)),  # Clamp to realistic range
        })

    return pd.DataFrame(markets)


def generate_team_stats_df(games_df: pd.DataFrame) -> pd.DataFrame:
    """Generate sample team stats dataframe."""
    team_stats = []

    for _, game in games_df.iterrows():
        # Home team stats
        team_stats.append({
            "game_id": game["game_id"],
            "team": game["home_team"],
            "is_home": True,
            "points_for": game["home_score"],
            "points_against": game["away_score"],
            "turnovers": np.random.randint(0, 4),
            "yards_for": np.random.randint(250, 450),
            "yards_against": np.random.randint(250, 450),
        })

        # Away team stats
        team_stats.append({
            "game_id": game["game_id"],
            "team": game["away_team"],
            "is_home": False,
            "points_for": game["away_score"],
            "points_against": game["home_score"],
            "turnovers": np.random.randint(0, 4),
            "yards_for": np.random.randint(250, 450),
            "yards_against": np.random.randint(250, 450),
        })

    return pd.DataFrame(team_stats)


def generate_games_markets_df(games_df: pd.DataFrame, markets_df: pd.DataFrame) -> pd.DataFrame:
    """Join games and markets dataframes."""
    return games_df.merge(markets_df[["game_id", "close_spread", "close_total"]], on="game_id", how="left")


def main():
    """Generate and save all sample data files."""
    output_dir = Path(__file__).parent.parent / "data" / "nfl" / "sample"
    staged_dir = output_dir / "staged"
    staged_dir.mkdir(parents=True, exist_ok=True)

    print("Generating sample data...")

    # Generate dataframes
    games_df = generate_games_df()
    print(f"Generated {len(games_df)} games")

    markets_df = generate_markets_df(games_df)
    print(f"Generated {len(markets_df)} market entries")

    team_stats_df = generate_team_stats_df(games_df)
    print(f"Generated {len(team_stats_df)} team stat entries")

    games_markets_df = generate_games_markets_df(games_df, markets_df)
    print(f"Generated {len(games_markets_df)} joined games-markets entries")

    # Save to parquet
    games_df.to_parquet(staged_dir / "games.parquet", index=False)
    print(f"Saved: {staged_dir / 'games.parquet'}")

    markets_df.to_parquet(staged_dir / "markets.parquet", index=False)
    print(f"Saved: {staged_dir / 'markets.parquet'}")

    team_stats_df.to_parquet(staged_dir / "team_stats.parquet", index=False)
    print(f"Saved: {staged_dir / 'team_stats.parquet'}")

    games_markets_df.to_parquet(staged_dir / "games_markets.parquet", index=False)
    print(f"Saved: {staged_dir / 'games_markets.parquet'}")

    print("\nSample data generation complete!")
    print(f"\nSummary:")
    print(f"  Teams: {SAMPLE_TEAMS}")
    print(f"  Season: {SAMPLE_SEASON}")
    print(f"  Weeks: {SAMPLE_WEEKS}")
    print(f"  Total games: {len(games_df)}")
    print(f"  Home win rate: {(games_df['home_score'] > games_df['away_score']).mean():.2%}")


if __name__ == "__main__":
    main()
