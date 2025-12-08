/**
 * QB Headshots Service
 * 
 * Loads and provides lookup functionality for quarterback headshot URLs
 * from the exported JSON file.
 * 
 * Handles both server-side (Node.js) and client-side (browser) contexts.
 */

export interface QBHeadshot {
    player_name: string;
    headshot_url: string;
    team: string;
}

let qbHeadshotsCache: QBHeadshot[] | null = null;
let qbHeadshotsMap: Map<string, string> | null = null;

/**
 * Check if we're running in a server-side context
 */
function isServerSide(): boolean {
    return typeof window === 'undefined';
}

/**
 * Load QB headshots from JSON file
 * Handles both server-side (filesystem) and client-side (fetch) contexts
 */
async function loadQBHeadshots(): Promise<QBHeadshot[]> {
    if (qbHeadshotsCache) {
        return qbHeadshotsCache;
    }

    try {
        let data: QBHeadshot[];
        const isServer = isServerSide();

        if (isServer) {
            // Server-side: read from filesystem using Node.js fs
            try {
                // Use require to avoid bundling fs in client code
                // eslint-disable-next-line @typescript-eslint/no-require-imports, @typescript-eslint/no-var-requires
                const fs = require('fs');
                // eslint-disable-next-line @typescript-eslint/no-require-imports, @typescript-eslint/no-var-requires
                const path = require('path');
                const filePath = path.join(process.cwd(), 'public', 'qb_images.json');
                
                if (!fs.existsSync(filePath)) {
                    console.warn(`QB headshots file not found at: ${filePath}`);
                    return [];
                }
                
                const fileContents = fs.readFileSync(filePath, 'utf-8');
                data = JSON.parse(fileContents);
            } catch (fsError: unknown) {
                // If fs fails, log and return empty array
                console.error('Failed to read QB headshots from filesystem:', fsError);
                return [];
            }
        } else {
            // Client-side: use fetch with relative URL
            // Double-check we're actually in browser
            if (typeof window === 'undefined' || typeof fetch === 'undefined') {
                console.warn('Cannot load QB headshots: not in browser context');
                return [];
            }
            
            try {
                const response = await fetch('/qb_images.json');
                if (!response.ok) {
                    console.warn('Failed to load QB headshots:', response.statusText);
                    return [];
                }
                data = await response.json();
            } catch (fetchError: unknown) {
                console.error('Failed to fetch QB headshots:', fetchError);
                return [];
            }
        }

        qbHeadshotsCache = data;
        return data;
    } catch (error) {
        console.error('Error loading QB headshots:', error);
        return [];
    }
}

/**
 * Build a lookup map for quick headshot URL retrieval
 */
function buildHeadshotMap(): Map<string, string> {
    if (qbHeadshotsMap) {
        return qbHeadshotsMap;
    }

    const map = new Map<string, string>();
    if (qbHeadshotsCache) {
        for (const qb of qbHeadshotsCache) {
            // Use player name as key (case-insensitive)
            const key = qb.player_name.toLowerCase().trim();
            if (qb.headshot_url) {
                map.set(key, qb.headshot_url);
            }
        }
    }
    qbHeadshotsMap = map;
    return map;
}

/**
 * Get headshot URL for a quarterback by name
 * 
 * @param playerName - Full name of the quarterback (e.g., "Patrick Mahomes")
 * @returns Headshot URL or empty string if not found
 */
export async function getQBHeadshot(playerName: string): Promise<string> {
    if (!playerName) {
        return '';
    }

    // Ensure headshots are loaded
    await loadQBHeadshots();
    
    // Build map if not already built
    const map = buildHeadshotMap();
    
    // Lookup by name (case-insensitive)
    const key = playerName.toLowerCase().trim();
    return map.get(key) || '';
}

/**
 * Get headshot URL synchronously (requires headshots to be pre-loaded)
 * 
 * @param playerName - Full name of the quarterback
 * @returns Headshot URL or empty string if not found
 */
export function getQBHeadshotSync(playerName: string): string {
    if (!playerName || !qbHeadshotsMap) {
        return '';
    }
    
    const key = playerName.toLowerCase().trim();
    return qbHeadshotsMap.get(key) || '';
}

/**
 * Preload QB headshots (call this early in app initialization)
 */
export async function preloadQBHeadshots(): Promise<void> {
    await loadQBHeadshots();
    buildHeadshotMap();
}

/**
 * Get all QB headshots
 */
export async function getAllQBHeadshots(): Promise<QBHeadshot[]> {
    return await loadQBHeadshots();
}

