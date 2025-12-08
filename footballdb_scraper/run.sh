#!/bin/bash
# Convenience script to run FootballDB scraper scripts
# Usage: ./run.sh check_progress
#        ./run.sh scrape_splits --year 2024 --priority-only
#        ./run.sh scrape_coaches

cd "$(dirname "$0")/.." || exit 1
source venv/bin/activate 2>/dev/null || true

SCRIPT=$1
shift

case $SCRIPT in
    check_progress|progress)
        python footballdb_scraper/scripts/check_progress.py "$@"
        ;;
    scrape_splits|splits)
        python footballdb_scraper/scripts/scrape_splits.py "$@"
        ;;
    scrape_coaches|coaches)
        python footballdb_scraper/scripts/scrape_coaches.py "$@"
        ;;
    export_features|features)
        python footballdb_scraper/scripts/export_features.py "$@"
        ;;
    *)
        echo "Usage: $0 {check_progress|scrape_splits|scrape_coaches|export_features} [args...]"
        echo ""
        echo "Examples:"
        echo "  $0 check_progress"
        echo "  $0 scrape_splits --year 2024 --priority-only"
        echo "  $0 scrape_coaches"
        exit 1
        ;;
esac



