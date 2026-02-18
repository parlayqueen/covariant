"""
ESPN Injury Data Scraper

Scrapes injury reports from ESPN for various sports.
"""

import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import re


@dataclass
class InjuryReport:
    """Player injury information"""
    player_name: str
    team: str
    position: str
    status: str  # Out, Questionable, Doubtful, Day-to-Day, Probable
    injury: str  # Body part/injury type
    date_updated: Optional[datetime] = None
    
    def severity_score(self) -> float:
        """
        Calculate injury severity (0-1, higher = worse).
        
        Returns:
            Severity score
        """
        severity_map = {
            'out': 1.0,
            'doubtful': 0.8,
            'questionable': 0.6,
            'day-to-day': 0.4,
            'day to day': 0.4,
            'probable': 0.2,
            'gtd': 0.6,  # Game-time decision
        }
        
        status_lower = self.status.lower()
        for key, value in severity_map.items():
            if key in status_lower:
                return value
        
        return 0.5  # Default if unknown


class ESPNInjuryClient:
    """
    Scrape injury data from ESPN.
    
    Features:
    - Get injuries by sport
    - Calculate team injury impact
    - Filter by team
    """
    
    INJURY_URLS = {
        'nba': 'https://www.espn.com/nba/injuries',
        'nfl': 'https://www.espn.com/nfl/injuries',
        'mlb': 'https://www.espn.com/mlb/injuries',
        'nhl': 'https://www.espn.com/nhl/injuries',
        'ncaaf': 'https://www.espn.com/college-football/injuries',
        'ncaab': 'https://www.espn.com/mens-college-basketball/injuries',
    }
    
    def __init__(self):
        """Initialize injury client"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_injuries(self, sport: str) -> List[InjuryReport]:
        """
        Scrape injuries for a sport.
        
        Args:
            sport: Sport key (nba, nfl, mlb, nhl, ncaaf, ncaab)
            
        Returns:
            List of InjuryReport objects
        """
        if sport.lower() not in self.INJURY_URLS:
            raise ValueError(f"Sport '{sport}' not supported. Options: {list(self.INJURY_URLS.keys())}")
        
        url = self.INJURY_URLS[sport.lower()]
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching injuries: {e}")
            return []
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        injuries = []
        
        # ESPN injury page structure varies, but generally has tables or sections per team
        # Try multiple parsing strategies
        
        # Strategy 1: Look for injury tables
        tables = soup.find_all('table', class_=re.compile('injuries|Table'))
        
        if tables:
            injuries.extend(self._parse_injury_tables(tables))
        else:
            # Strategy 2: Look for structured divs
            injury_sections = soup.find_all('div', class_=re.compile('injury|player'))
            if injury_sections:
                injuries.extend(self._parse_injury_divs(injury_sections))
        
        return injuries
    
    def _parse_injury_tables(self, tables: List) -> List[InjuryReport]:
        """Parse injury data from HTML tables"""
        injuries = []
        current_team = None
        
        for table in tables:
            # Look for team name (often in caption or preceding heading)
            caption = table.find('caption')
            if caption:
                current_team = caption.get_text(strip=True)
            
            # If no caption, look for preceding h2/h3
            if not current_team:
                prev_heading = table.find_previous(['h2', 'h3', 'h4'])
                if prev_heading:
                    current_team = prev_heading.get_text(strip=True)
            
            rows = table.find_all('tr')
            
            for row in rows[1:]:  # Skip header
                cols = row.find_all(['td', 'th'])
                
                if len(cols) >= 3:
                    try:
                        player_name = cols[0].get_text(strip=True)
                        
                        # Skip if it's a header row
                        if player_name.lower() in ['name', 'player', 'team']:
                            continue
                        
                        position = cols[1].get_text(strip=True) if len(cols) > 1 else ''
                        status = cols[2].get_text(strip=True) if len(cols) > 2 else ''
                        injury = cols[3].get_text(strip=True) if len(cols) > 3 else ''
                        
                        # Clean up data
                        player_name = self._clean_player_name(player_name)
                        
                        if player_name and status:
                            injuries.append(InjuryReport(
                                player_name=player_name,
                                team=current_team or 'Unknown',
                                position=position,
                                status=status,
                                injury=injury,
                                date_updated=datetime.now()
                            ))
                    except Exception as e:
                        continue
        
        return injuries
    
    def _parse_injury_divs(self, divs: List) -> List[InjuryReport]:
        """Parse injury data from div structure"""
        injuries = []
        # Implement if table parsing fails
        # This is a fallback for different ESPN layouts
        return injuries
    
    def _clean_player_name(self, name: str) -> str:
        """Clean player name of extra characters"""
        # Remove position abbreviations in parentheses
        name = re.sub(r'\([A-Z]{1,3}\)', '', name)
        # Remove extra whitespace
        name = ' '.join(name.split())
        return name.strip()
    
    def get_team_injuries(
        self,
        sport: str,
        team_name: str,
        fuzzy_match: bool = True
    ) -> List[InjuryReport]:
        """
        Get injuries for a specific team.
        
        Args:
            sport: Sport key
            team_name: Team name to filter
            fuzzy_match: Allow partial team name matches
            
        Returns:
            List of injuries for that team
        """
        all_injuries = self.get_injuries(sport)
        
        if fuzzy_match:
            # Normalize team names for comparison
            team_name_lower = team_name.lower()
            return [
                inj for inj in all_injuries
                if team_name_lower in inj.team.lower() or
                   inj.team.lower() in team_name_lower
            ]
        else:
            return [inj for inj in all_injuries if inj.team == team_name]
    
    def calculate_team_impact(
        self,
        sport: str,
        team_name: str
    ) -> Dict:
        """
        Calculate overall injury impact for a team.
        
        Args:
            sport: Sport key
            team_name: Team name
            
        Returns:
            Dictionary with impact metrics
        """
        injuries = self.get_team_injuries(sport, team_name)
        
        if not injuries:
            return {
                'team': team_name,
                'total_injuries': 0,
                'avg_severity': 0.0,
                'uncertainty_score': 0.0,
                'key_injuries': []
            }
        
        # Calculate metrics
        total = len(injuries)
        avg_severity = sum(inj.severity_score() for inj in injuries) / total
        
        # Uncertainty increases with number and severity
        uncertainty_score = min(avg_severity * (1 + total * 0.1), 1.0)
        
        # Identify key injuries (starters/stars) - this is heuristic
        key_injuries = [
            inj for inj in injuries
            if inj.severity_score() >= 0.6  # Questionable or worse
        ]
        
        return {
            'team': team_name,
            'total_injuries': total,
            'avg_severity': avg_severity,
            'uncertainty_score': uncertainty_score,
            'key_injuries': key_injuries,
            'all_injuries': injuries
        }
    
    def format_injury_report(
        self,
        sport: str,
        team_name: Optional[str] = None
    ) -> str:
        """
        Create formatted injury report.
        
        Args:
            sport: Sport key
            team_name: Optional team filter
            
        Returns:
            Formatted string
        """
        if team_name:
            injuries = self.get_team_injuries(sport, team_name)
            title = f"Injury Report: {team_name}"
        else:
            injuries = self.get_injuries(sport)
            title = f"Injury Report: {sport.upper()}"
        
        if not injuries:
            return f"{title}\nNo injuries reported"
        
        lines = [title, "=" * 70]
        
        # Group by team
        by_team = {}
        for inj in injuries:
            if inj.team not in by_team:
                by_team[inj.team] = []
            by_team[inj.team].append(inj)
        
        for team, team_injuries in sorted(by_team.items()):
            lines.append(f"\n{team}:")
            for inj in team_injuries:
                severity = "🔴" if inj.severity_score() >= 0.8 else \
                          "🟡" if inj.severity_score() >= 0.4 else "🟢"
                lines.append(
                    f"  {severity} {inj.player_name} ({inj.position}) - "
                    f"{inj.status} - {inj.injury}"
                )
        
        return "\n".join(lines)


# Fallback: JSON API approach (if available)
class ESPNAPIClient:
    """
    ESPN API client (unofficial).
    
    Note: ESPN's public APIs are limited and undocumented.
    This is a best-effort implementation.
    """
    
    BASE_URL = "https://site.api.espn.com/apis/site/v2/sports"
    
    def __init__(self):
        self.session = requests.Session()
    
    def get_scoreboard(self, sport: str, league: str) -> Dict:
        """
        Get scoreboard data (includes injury notes sometimes).
        
        Args:
            sport: Sport type (basketball, football, etc.)
            league: League (nba, nfl, etc.)
            
        Returns:
            Scoreboard JSON
        """
        url = f"{self.BASE_URL}/{sport}/{league}/scoreboard"
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching scoreboard: {e}")
            return {}
    
    def extract_injury_notes(self, scoreboard_data: Dict) -> List[str]:
        """Extract injury-related notes from scoreboard"""
        notes = []
        
        events = scoreboard_data.get('events', [])
        
        for event in events:
            competitions = event.get('competitions', [])
            for comp in competitions:
                # Check notes
                if 'notes' in comp:
                    for note in comp['notes']:
                        if 'injury' in note.get('headline', '').lower():
                            notes.append(note.get('headline'))
        
        return notes


# Example usage
if __name__ == "__main__":
    client = ESPNInjuryClient()
    
    # Get NBA injuries
    print("Fetching NBA injuries...")
    injuries = client.get_injuries('nba')
    
    print(f"\nFound {len(injuries)} total injuries")
    
    # Show sample
    if injuries:
        print("\nSample injuries:")
        for inj in injuries[:5]:
            print(f"  {inj.player_name} ({inj.team}) - {inj.status} - {inj.injury}")
    
    # Team-specific
    print("\n" + "="*70)
    print("Team-specific injury impact:")
    
    # Get unique teams
    teams = list(set(inj.team for inj in injuries))
    
    if teams:
        sample_team = teams[0]
        impact = client.calculate_team_impact('nba', sample_team)
        
        print(f"\n{impact['team']}:")
        print(f"  Total Injuries: {impact['total_injuries']}")
        print(f"  Avg Severity: {impact['avg_severity']:.2f}")
        print(f"  Uncertainty Score: {impact['uncertainty_score']:.2f}")
        print(f"  Key Injuries: {len(impact['key_injuries'])}")
