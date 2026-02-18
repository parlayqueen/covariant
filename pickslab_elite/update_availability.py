#!/usr/bin/env python3
"""
CLI Tool: Update Availability

Updates player availability data from ESPN for a specific date.

Usage:
    python -m pickslab_pro.update_availability --league nba --date 2024-01-15
"""

import argparse
import logging
from datetime import date, datetime

from pickslab_elite.adapters.availability.espn_nba import update_nba_availability


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Update player availability from ESPN'
    )
    
    parser.add_argument(
        '--league',
        type=str,
        required=True,
        choices=['nba', 'nfl', 'nhl', 'mlb'],
        help='League to update'
    )
    
    parser.add_argument(
        '--date',
        type=str,
        default=None,
        help='Date to update (YYYY-MM-DD), defaults to today'
    )
    
    args = parser.parse_args()
    
    # Parse date
    if args.date:
        target_date = datetime.strptime(args.date, '%Y-%m-%d').date()
    else:
        target_date = date.today()
    
    logger.info(f"Updating {args.league.upper()} availability for {target_date}")
    
    # Update based on league
    if args.league == 'nba':
        stats = update_nba_availability(target_date)
    else:
        logger.error(f"League {args.league} not yet implemented")
        return 1
    
    # Print summary
    print("\n" + "="*60)
    print(f"AVAILABILITY UPDATE COMPLETE: {args.league.upper()}")
    print("="*60)
    print(f"Date: {target_date}")
    print(f"Games Found: {stats['games_found']}")
    print(f"Players Updated: {stats['players_updated']}")
    print(f"Statuses Updated: {stats['statuses_updated']}")
    print(f"Errors: {len(stats['errors'])}")
    
    if stats['errors']:
        print("\nErrors:")
        for error in stats['errors'][:5]:  # Show first 5
            print(f"  - {error}")
    
    print("="*60)
    
    return 0 if not stats['errors'] else 1


if __name__ == '__main__':
    exit(main())
