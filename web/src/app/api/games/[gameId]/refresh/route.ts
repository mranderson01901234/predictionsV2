import { NextRequest, NextResponse } from 'next/server';

interface GameData {
  score: { home: number; away: number };
  quarter?: number;
  timeRemaining?: string;
  possession?: string;
  stats: Record<string, unknown>;
  injuries: Array<{ playerId: string; playerName: string; status: string }>;
  lastUpdated: string;
}

// In-memory cache (replace with Redis/KV in production)
const gameCache = new Map<string, GameData>();

/**
 * Fetch fresh game data from external sources
 * This is a placeholder that would call real data sources
 */
async function fetchFreshGameData(gameId: string): Promise<GameData> {
  // TODO: Implement actual scraping/API calls based on game status
  // This would typically call ESPN API, NFL API, or scrape data

  // For now, return mock data with a small random variation to simulate changes
  const cached = gameCache.get(gameId);
  const hasChange = Math.random() > 0.7; // 30% chance of change

  return {
    score: {
      home: cached?.score.home ?? 24,
      away: cached?.score.away ?? 17,
    },
    quarter: 3,
    timeRemaining: '8:45',
    possession: 'home',
    stats: {},
    injuries: [],
    lastUpdated: new Date().toISOString(),
  };
}

/**
 * Detect changes between old and new game data
 */
function detectChanges(oldData: GameData | undefined, newData: GameData): string[] {
  if (!oldData) return ['Initial data load'];

  const changes: string[] = [];

  // Score change
  if (oldData.score.home !== newData.score.home || oldData.score.away !== newData.score.away) {
    changes.push(`Score updated: ${newData.score.away}-${newData.score.home}`);
  }

  // Quarter change
  if (oldData.quarter !== newData.quarter) {
    changes.push(`Now in Q${newData.quarter}`);
  }

  // Possession change
  if (oldData.possession !== newData.possession) {
    changes.push(`Possession changed to ${newData.possession}`);
  }

  // Injury changes
  const oldInjuries = new Set(oldData.injuries.map(i => i.playerId));
  const newInjuries = newData.injuries.filter(i => !oldInjuries.has(i.playerId));
  if (newInjuries.length > 0) {
    changes.push(`Injury update: ${newInjuries.map(i => i.playerName).join(', ')}`);
  }

  return changes;
}

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ gameId: string }> }
) {
  const { gameId } = await params;

  try {
    // Fetch fresh data
    const freshData = await fetchFreshGameData(gameId);

    // Compare with cached data
    const cachedData = gameCache.get(gameId);
    const changes = detectChanges(cachedData, freshData);

    // Update cache
    gameCache.set(gameId, freshData);

    return NextResponse.json({
      success: true,
      changesDetected: changes.length > 0,
      changes,
      data: freshData,
      lastUpdated: freshData.lastUpdated,
    });
  } catch (error) {
    console.error('Refresh error:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to refresh data' },
      { status: 500 }
    );
  }
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ gameId: string }> }
) {
  const { gameId } = await params;

  const cachedData = gameCache.get(gameId);

  if (!cachedData) {
    return NextResponse.json(
      { success: false, error: 'No cached data available' },
      { status: 404 }
    );
  }

  return NextResponse.json({
    success: true,
    data: cachedData,
    lastUpdated: cachedData.lastUpdated,
  });
}
