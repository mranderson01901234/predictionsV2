/**
 * QB Headshot Image URLs
 *
 * Uses ESPN CDN for high-quality, publicly accessible headshots
 * Format: https://a.espncdn.com/i/headshots/nfl/players/full/{playerId}.png
 * Falls back to initials if no image found
 */

// Map of QB names to their ESPN headshot URLs
// ESPN player IDs are stable and publicly accessible
const QB_IMAGES: Record<string, string> = {
  // AFC East
  'Josh Allen': 'https://a.espncdn.com/i/headshots/nfl/players/full/3918298.png',
  'Tua Tagovailoa': 'https://a.espncdn.com/i/headshots/nfl/players/full/4241479.png',
  'Aaron Rodgers': 'https://a.espncdn.com/i/headshots/nfl/players/full/8439.png',
  'Mac Jones': 'https://a.espncdn.com/i/headshots/nfl/players/full/4361418.png',
  'Drake Maye': 'https://a.espncdn.com/i/headshots/nfl/players/full/4686922.png',

  // AFC North
  'Lamar Jackson': 'https://a.espncdn.com/i/headshots/nfl/players/full/3916387.png',
  'Joe Burrow': 'https://a.espncdn.com/i/headshots/nfl/players/full/3915511.png',
  'Deshaun Watson': 'https://a.espncdn.com/i/headshots/nfl/players/full/2969939.png',
  'Russell Wilson': 'https://a.espncdn.com/i/headshots/nfl/players/full/14881.png',
  'Justin Fields': 'https://a.espncdn.com/i/headshots/nfl/players/full/4362887.png',

  // AFC South
  'C.J. Stroud': 'https://a.espncdn.com/i/headshots/nfl/players/full/4432577.png',
  'Anthony Richardson': 'https://a.espncdn.com/i/headshots/nfl/players/full/4569618.png',
  'Trevor Lawrence': 'https://a.espncdn.com/i/headshots/nfl/players/full/4360310.png',
  'Will Levis': 'https://a.espncdn.com/i/headshots/nfl/players/full/4362238.png',

  // AFC West
  'Patrick Mahomes': 'https://a.espncdn.com/i/headshots/nfl/players/full/3139477.png',
  'Justin Herbert': 'https://a.espncdn.com/i/headshots/nfl/players/full/4038941.png',
  'Bo Nix': 'https://a.espncdn.com/i/headshots/nfl/players/full/4426385.png',
  'Aidan O\'Connell': 'https://a.espncdn.com/i/headshots/nfl/players/full/4259545.png',
  'Gardner Minshew': 'https://a.espncdn.com/i/headshots/nfl/players/full/3122840.png',

  // NFC East
  'Jalen Hurts': 'https://a.espncdn.com/i/headshots/nfl/players/full/4040715.png',
  'Dak Prescott': 'https://a.espncdn.com/i/headshots/nfl/players/full/2577417.png',
  'Daniel Jones': 'https://a.espncdn.com/i/headshots/nfl/players/full/3917315.png',
  'Jayden Daniels': 'https://a.espncdn.com/i/headshots/nfl/players/full/4429022.png',

  // NFC North
  'Jordan Love': 'https://a.espncdn.com/i/headshots/nfl/players/full/4036378.png',
  'Jared Goff': 'https://a.espncdn.com/i/headshots/nfl/players/full/3046779.png',
  'Caleb Williams': 'https://a.espncdn.com/i/headshots/nfl/players/full/4432080.png',
  'Sam Darnold': 'https://a.espncdn.com/i/headshots/nfl/players/full/3912547.png',
  'J.J. McCarthy': 'https://a.espncdn.com/i/headshots/nfl/players/full/4429013.png',

  // NFC South
  'Baker Mayfield': 'https://a.espncdn.com/i/headshots/nfl/players/full/3052587.png',
  'Derek Carr': 'https://a.espncdn.com/i/headshots/nfl/players/full/16757.png',
  'Bryce Young': 'https://a.espncdn.com/i/headshots/nfl/players/full/4432178.png',
  'Kirk Cousins': 'https://a.espncdn.com/i/headshots/nfl/players/full/14880.png',
  'Michael Penix Jr.': 'https://a.espncdn.com/i/headshots/nfl/players/full/4362530.png',

  // NFC West
  'Brock Purdy': 'https://a.espncdn.com/i/headshots/nfl/players/full/4361741.png',
  'Kyler Murray': 'https://a.espncdn.com/i/headshots/nfl/players/full/3917792.png',
  'Geno Smith': 'https://a.espncdn.com/i/headshots/nfl/players/full/14875.png',
  'Matthew Stafford': 'https://a.espncdn.com/i/headshots/nfl/players/full/12483.png',
};

// Normalized name lookup (handles variations like "Pat Mahomes" -> "Patrick Mahomes")
const NAME_ALIASES: Record<string, string> = {
  'Pat Mahomes': 'Patrick Mahomes',
  'Patty Mahomes': 'Patrick Mahomes',
  'T.Lawrence': 'Trevor Lawrence',
  'T. Lawrence': 'Trevor Lawrence',
  'TLaw': 'Trevor Lawrence',
  'A. Richardson': 'Anthony Richardson',
  'AR': 'Anthony Richardson',
  'Josh': 'Josh Allen',
  'Lamar': 'Lamar Jackson',
  'Burrow': 'Joe Burrow',
  'Joey B': 'Joe Burrow',
  'Tua': 'Tua Tagovailoa',
  'Dak': 'Dak Prescott',
  'Hurts': 'Jalen Hurts',
  'Stroud': 'C.J. Stroud',
  'CJ Stroud': 'C.J. Stroud',
  'CJ': 'C.J. Stroud',
  'Herbert': 'Justin Herbert',
  'Goff': 'Jared Goff',
  'Purdy': 'Brock Purdy',
  'Love': 'Jordan Love',
  'Caleb': 'Caleb Williams',
};

/**
 * Get QB headshot image URL by name
 * Returns null if not found
 */
export function getQBImage(name: string): string | null {
  if (!name) return null;

  // Try direct lookup first
  if (QB_IMAGES[name]) {
    return QB_IMAGES[name];
  }

  // Try alias lookup
  const aliasedName = NAME_ALIASES[name];
  if (aliasedName && QB_IMAGES[aliasedName]) {
    return QB_IMAGES[aliasedName];
  }

  // Try partial match (last name)
  const lastName = name.split(' ').pop()?.toLowerCase();
  if (lastName) {
    for (const [qbName, url] of Object.entries(QB_IMAGES)) {
      if (qbName.toLowerCase().endsWith(lastName)) {
        return url;
      }
    }
  }

  return null;
}

/**
 * Get QB initials for fallback display
 */
export function getQBInitials(name: string): string {
  if (!name) return '?';
  return name
    .split(' ')
    .map(n => n[0])
    .join('')
    .toUpperCase()
    .slice(0, 2);
}

/**
 * Check if we have an image for a QB
 */
export function hasQBImage(name: string): boolean {
  return getQBImage(name) !== null;
}
