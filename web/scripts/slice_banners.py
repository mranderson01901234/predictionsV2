from PIL import Image
import os

# Grid configuration
COLS = 6
ROWS = 7

# Team mapping: (row, col) -> team abbreviation
# Reading from your image, left to right, top to bottom
# Set None for duplicates to skip them

TEAM_GRID = {
    # Row 0: BUF, BAL, HOU, LV, DAL, ATL
    (0, 0): "BUF",
    (0, 1): "BAL",
    (0, 2): "HOU",
    (0, 3): "LV",
    (0, 4): "DAL",
    (0, 5): "ATL",
    
    # Row 1: MIA, CIN, IND, DEN, NYG, WAS
    (1, 0): "MIA",
    (1, 1): "CIN",
    (1, 2): "IND",
    (1, 3): "DEN",
    (1, 4): "NYG",
    (1, 5): "WAS",
    
    # Row 2: NE, CLE, JAX, KC, DET, GB
    (2, 0): "NE",
    (2, 1): "CLE",
    (2, 2): "JAX",
    (2, 3): "KC",
    (2, 4): "DET",
    (2, 5): "GB",
    
    # Row 3: NYJ, PIT, TEN, LV(dup), CHI, TB
    (3, 0): "NYJ",
    (3, 1): "PIT",
    (3, 2): "TEN",
    (3, 3): None,  # LV duplicate
    (3, 4): "CHI",
    (3, 5): "TB",
    
    # Row 4: BAL(dup), HOU(dup), DEN(dup), LAC, CAR, NO
    (4, 0): None,  # BAL duplicate
    (4, 1): None,  # HOU duplicate
    (4, 2): None,  # DEN duplicate
    (4, 3): "LAC",
    (4, 4): "CAR",
    (4, 5): "NO",
    
    # Row 5: CIN(dup), PHI, ATL(dup), MIN, NO(dup), TB(dup)
    (5, 0): None,  # CIN duplicate
    (5, 1): "PHI",
    (5, 2): None,  # ATL duplicate
    (5, 3): "MIN",
    (5, 4): None,  # NO duplicate
    (5, 5): None,  # TB duplicate
    
    # Row 6: CLE(dup), WAS(dup), ARI, LAR, SF, SEA
    (6, 0): None,  # CLE duplicate
    (6, 1): None,  # WAS duplicate
    (6, 2): "ARI",
    (6, 3): "LAR",
    (6, 4): "SF",
    (6, 5): "SEA",
}


def slice_banners(image_path: str, output_dir: str = "./banners"):
    """Slice the team banner grid into individual images."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    img = Image.open(image_path)
    width, height = img.size
    
    cell_width = width // COLS
    cell_height = height // ROWS
    
    print(f"Image size: {width}x{height}")
    print(f"Cell size: {cell_width}x{cell_height}")
    
    saved_count = 0
    for (row, col), team in TEAM_GRID.items():
        if team is None:
            continue
            
        left = col * cell_width
        top = row * cell_height
        right = left + cell_width
        bottom = top + cell_height
        
        cell = img.crop((left, top, right, bottom))
        output_path = os.path.join(output_dir, f"{team.lower()}.png")
        cell.save(output_path, "PNG")
        print(f"Saved: {output_path}")
        saved_count += 1
    
    print(f"\nDone! Saved {saved_count} team banners to {output_dir}")
    
    # Verify we got all 32 teams
    expected_teams = {"BUF", "MIA", "NE", "NYJ", "BAL", "CIN", "CLE", "PIT", 
                      "HOU", "IND", "JAX", "TEN", "DEN", "KC", "LV", "LAC",
                      "DAL", "NYG", "PHI", "WAS", "CHI", "DET", "GB", "MIN",
                      "ATL", "CAR", "NO", "TB", "ARI", "LAR", "SF", "SEA"}
    saved_teams = {t for t in TEAM_GRID.values() if t}
    missing = expected_teams - saved_teams
    if missing:
        print(f"\n⚠️  Missing teams: {missing}")
    else:
        print("\n✓ All 32 teams captured!")


if __name__ == "__main__":
    # Get the script directory and construct paths relative to it
    script_dir = os.path.dirname(os.path.abspath(__file__))
    web_dir = os.path.dirname(script_dir)
    image_path = os.path.join(web_dir, "images", "flags.jpeg")
    output_dir = os.path.join(web_dir, "images", "banners")
    
    print(f"Input image: {image_path}")
    print(f"Output directory: {output_dir}\n")
    
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found at {image_path}")
        exit(1)
    
    slice_banners(image_path, output_dir)

