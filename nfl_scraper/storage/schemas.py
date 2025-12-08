"""
Data schemas for scraped NFL.com data.
"""

from dataclasses import dataclass
from typing import Optional, List
from datetime import date


@dataclass
class InjuryRecord:
    """Single injury record for a player in a specific week."""
    season: int
    week: int
    game_date: Optional[date]
    team: str
    opponent: str
    player_name: str
    player_id: str  # NFL.com player slug
    position: str
    injury_type: Optional[str]  # "Hamstring", "Knee", "Ankle", etc.
    injury_types: List[str]  # Can have multiple: ["Toe", "Ankle"]
    practice_status_wed: Optional[str]  # "DNP", "Limited", "Full", None
    practice_status_thu: Optional[str]
    practice_status_fri: Optional[str]
    practice_status_final: str  # Final practice status before game
    game_status: Optional[str]  # "Out", "Doubtful", "Questionable", None
    is_resting: bool  # "Not injury related - resting player"
    
    def to_dict(self) -> dict:
        return {
            'season': self.season,
            'week': self.week,
            'game_date': self.game_date.isoformat() if self.game_date else None,
            'team': self.team,
            'opponent': self.opponent,
            'player_name': self.player_name,
            'player_id': self.player_id,
            'position': self.position,
            'injury_type': self.injury_type,
            'injury_types': self.injury_types,
            'practice_status_wed': self.practice_status_wed,
            'practice_status_thu': self.practice_status_thu,
            'practice_status_fri': self.practice_status_fri,
            'practice_status_final': self.practice_status_final,
            'game_status': self.game_status,
            'is_resting': self.is_resting,
        }


@dataclass
class PlayerSituationalStats:
    """Player situational stats for a specific season."""
    player_id: str
    player_name: str
    position: str
    team: str
    season: int
    
    # By Quarter
    q1_attempts: int
    q1_completions: int
    q1_yards: int
    q1_tds: int
    q1_ints: int
    q1_rating: float
    
    q2_attempts: int
    q2_completions: int
    q2_yards: int
    q2_tds: int
    q2_ints: int
    q2_rating: float
    
    q3_attempts: int
    q3_completions: int
    q3_yards: int
    q3_tds: int
    q3_ints: int
    q3_rating: float
    
    q4_attempts: int
    q4_completions: int
    q4_yards: int
    q4_tds: int
    q4_ints: int
    q4_rating: float
    
    q4_within_7_attempts: int  # 4th quarter, game within 7 points
    q4_within_7_completions: int
    q4_within_7_yards: int
    q4_within_7_tds: int
    q4_within_7_ints: int
    q4_within_7_rating: float
    
    # By Point Differential
    ahead_attempts: int
    ahead_completions: int
    ahead_yards: int
    ahead_tds: int
    ahead_ints: int
    ahead_rating: float
    
    behind_attempts: int
    behind_completions: int
    behind_yards: int
    behind_tds: int
    behind_ints: int
    behind_rating: float
    
    behind_1_8_attempts: int  # Behind by 1-8 points
    behind_1_8_completions: int
    behind_1_8_yards: int
    behind_1_8_tds: int
    behind_1_8_ints: int
    behind_1_8_rating: float
    
    tied_attempts: int
    tied_completions: int
    tied_yards: int
    tied_tds: int
    tied_ints: int
    tied_rating: float
    
    # Home vs Away
    home_attempts: int
    home_completions: int
    home_yards: int
    home_tds: int
    home_ints: int
    home_rating: float
    
    away_attempts: int
    away_completions: int
    away_yards: int
    away_tds: int
    away_ints: int
    away_rating: float
    
    # By Half
    first_half_attempts: int
    first_half_completions: int
    first_half_yards: int
    first_half_tds: int
    first_half_ints: int
    first_half_rating: float
    
    second_half_attempts: int
    second_half_completions: int
    second_half_yards: int
    second_half_tds: int
    second_half_ints: int
    second_half_rating: float
    
    # Red Zone (Field Position)
    red_zone_attempts: int  # Opp 19-1 yard line
    red_zone_completions: int
    red_zone_yards: int
    red_zone_tds: int
    red_zone_ints: int
    red_zone_rating: float


@dataclass
class PlayerCareerStats:
    """Player career stats summary."""
    player_id: str
    player_name: str
    position: str
    
    # Career totals (for QBs)
    career_games: int
    career_games_started: int
    career_completions: int
    career_attempts: int
    career_yards: int
    career_tds: int
    career_ints: int
    career_sacks: int
    career_rating: float
    
    # Career rushing (for mobile QBs)
    career_rush_attempts: int
    career_rush_yards: int
    career_rush_tds: int
    career_fumbles: int
    career_fumbles_lost: int
    
    # Season-by-season breakdown
    seasons: List[dict]  # List of season stats


@dataclass 
class TransactionRecord:
    """Single transaction record."""
    date: date
    transaction_type: str  # "Trade", "Signing", "Waiver", "IR", etc.
    from_team: Optional[str]
    to_team: Optional[str]
    player_name: str
    player_id: str
    position: Optional[str]
    details: Optional[str]

