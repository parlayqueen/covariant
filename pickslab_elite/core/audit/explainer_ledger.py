"""
Explainer Ledger

Audit trail for availability adjustments.

Stores WHY each adjustment was made for debugging and validation.
"""

import json
import logging
from pathlib import Path
from typing import Dict
from datetime import datetime


logger = logging.getLogger(__name__)


class ExplainerLedger:
    """
    Persistent audit log of availability adjustments.
    
    Stores explanations as JSONL for easy querying.
    """
    
    def __init__(self, log_dir: str = "data/audit"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.log_dir / f"availability_adjustments.jsonl"
    
    def log_adjustment(self, explanation: Dict):
        """
        Log an adjustment explanation.
        
        Args:
            explanation: Dict from apply_availability_adjustments()
        """
        # Add metadata
        explanation['logged_at'] = datetime.utcnow().isoformat()
        
        # Append to JSONL
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(explanation) + '\n')
        
        logger.debug(f"Logged adjustment for game {explanation.get('game_id')}")
    
    def query_game(self, game_id: str) -> List[Dict]:
        """Query explanations for a game"""
        if not self.log_file.exists():
            return []
        
        results = []
        
        with open(self.log_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get('game_id') == game_id:
                        results.append(entry)
                except:
                    continue
        
        return results
    
    def get_latest(self, n: int = 10) -> List[Dict]:
        """Get latest N explanations"""
        if not self.log_file.exists():
            return []
        
        all_entries = []
        
        with open(self.log_file, 'r') as f:
            for line in f:
                try:
                    all_entries.append(json.loads(line))
                except:
                    continue
        
        return all_entries[-n:]


# Singleton
_ledger = None


def get_explainer_ledger() -> ExplainerLedger:
    """Get singleton ledger"""
    global _ledger
    if _ledger is None:
        _ledger = ExplainerLedger()
    return _ledger
