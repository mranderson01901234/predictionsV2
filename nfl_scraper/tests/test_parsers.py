"""
Unit tests for NFL.com parsers.
"""

import unittest
from bs4 import BeautifulSoup

from parsers.injury_parser import InjuryParser


class TestInjuryParser(unittest.TestCase):
    """Test injury parser functionality."""
    
    def setUp(self):
        self.parser = InjuryParser()
    
    def test_parse_injury_types(self):
        """Test parsing injury types from text."""
        # Single injury
        injuries, is_resting = self.parser._parse_injury_types("Hamstring")
        self.assertEqual(injuries, ["Hamstring"])
        self.assertFalse(is_resting)
        
        # Multiple injuries
        injuries, is_resting = self.parser._parse_injury_types("Knee, Ankle")
        self.assertEqual(injuries, ["Knee", "Ankle"])
        self.assertFalse(is_resting)
        
        # Resting player
        injuries, is_resting = self.parser._parse_injury_types("Not injury related - resting player")
        self.assertTrue(is_resting)
        
        # Injury with resting note
        injuries, is_resting = self.parser._parse_injury_types("Shoulder, Not injury related - resting player")
        self.assertEqual(injuries, ["Shoulder"])
        self.assertTrue(is_resting)
    
    def test_normalize_practice_status(self):
        """Test practice status normalization."""
        self.assertEqual(self.parser._normalize_practice_status("Full Participation in Practice"), "Full")
        self.assertEqual(self.parser._normalize_practice_status("Limited Participation in Practice"), "Limited")
        self.assertEqual(self.parser._normalize_practice_status("Did Not Participate in Practice"), "DNP")
    
    def test_normalize_game_status(self):
        """Test game status normalization."""
        self.assertEqual(self.parser._normalize_game_status("Out"), "Out")
        self.assertEqual(self.parser._normalize_game_status("Doubtful"), "Doubtful")
        self.assertEqual(self.parser._normalize_game_status("Questionable"), "Questionable")
        self.assertIsNone(self.parser._normalize_game_status(""))
    
    def test_extract_player_id(self):
        """Test extracting player ID from URL."""
        self.assertEqual(
            self.parser._extract_player_id("/players/patrick-mahomes/"),
            "patrick-mahomes"
        )
        self.assertEqual(
            self.parser._extract_player_id("/players/josh-allen/stats/"),
            "josh-allen"
        )
    
    def test_slug_to_abbr(self):
        """Test team slug to abbreviation conversion."""
        self.assertEqual(self.parser._slug_to_abbr("kansas-city-chiefs"), "KC")
        self.assertEqual(self.parser._slug_to_abbr("dallas-cowboys"), "DAL")
        self.assertEqual(self.parser._slug_to_abbr("new-york-giants"), "NYG")


if __name__ == '__main__':
    unittest.main()

