"""
Check scraping progress.

Shows:
- Files created
- File sizes
- Scraper processes running
"""
import subprocess
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_progress():
    """Check scraping progress."""
    # Data is saved relative to project root
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "raw" / "footballdb" / "player_splits"
    
    if not data_dir.exists():
        print("Data directory not found")
        return
    
    files = list(data_dir.glob("*.parquet"))
    
    if not files:
        print("No parquet files found yet")
        return
    
    print("=" * 60)
    print("SCRAPING PROGRESS")
    print("=" * 60)
    print(f"\nFiles created: {len(files)}")
    print("\nFiles:")
    
    total_size = 0
    for f in sorted(files):
        size = f.stat().st_size
        total_size += size
        size_mb = size / (1024 * 1024)
        print(f"  {f.name:30} {size_mb:8.2f} MB")
    
    print(f"\nTotal size: {total_size / (1024 * 1024):.2f} MB")
    
    # Check running processes
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True
        )
        scrapers = [line for line in result.stdout.split('\n') if 'scrape_splits' in line and 'grep' not in line]
        
        if scrapers:
            print(f"\nActive scrapers: {len(scrapers)}")
            for scraper in scrapers[:3]:  # Show first 3
                parts = scraper.split()
                if len(parts) > 10:
                    cmd = ' '.join(parts[10:])
                    print(f"  {cmd[:80]}")
        else:
            print("\nNo active scrapers found")
    except Exception as e:
        print(f"\nCould not check processes: {e}")

if __name__ == "__main__":
    check_progress()

