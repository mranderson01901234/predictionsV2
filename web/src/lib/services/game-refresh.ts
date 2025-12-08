/**
 * Game Data Refresh Service
 *
 * Handles background data refreshes with cooldowns
 * Triggered by natural language refresh intents
 */

import { RefreshResult } from '../ai/types';

const REFRESH_COOLDOWN_MS = 30_000; // 30 seconds
const refreshTimestamps = new Map<string, number>();

/**
 * Check if a game can be refreshed (not in cooldown)
 */
export function canRefresh(gameId: string): boolean {
  const lastRefresh = refreshTimestamps.get(gameId) || 0;
  return Date.now() - lastRefresh > REFRESH_COOLDOWN_MS;
}

/**
 * Get remaining cooldown time in milliseconds
 */
export function getCooldownRemaining(gameId: string): number {
  const lastRefresh = refreshTimestamps.get(gameId) || 0;
  const elapsed = Date.now() - lastRefresh;
  return Math.max(0, REFRESH_COOLDOWN_MS - elapsed);
}

/**
 * Refresh game data with cooldown protection
 */
export async function refreshGameData(gameId: string): Promise<RefreshResult> {
  // Check cooldown
  if (!canRefresh(gameId)) {
    const remaining = Math.ceil(getCooldownRemaining(gameId) / 1000);
    return {
      refreshed: false,
      changesDetected: false,
      error: `Just checked ${remaining} seconds ago. Data should still be fresh.`,
      lastUpdated: new Date(refreshTimestamps.get(gameId) || Date.now()),
    };
  }

  try {
    // Call internal refresh endpoint
    const response = await fetch(`/api/games/${gameId}/refresh`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });

    if (!response.ok) {
      throw new Error(`Refresh failed: ${response.status}`);
    }

    const data = await response.json();

    // Update timestamp
    refreshTimestamps.set(gameId, Date.now());

    return {
      refreshed: true,
      changesDetected: data.changesDetected || false,
      changes: data.changes || [],
      lastUpdated: new Date(),
    };
  } catch (error) {
    console.error('Game refresh error:', error);
    return {
      refreshed: false,
      changesDetected: false,
      error: 'Trouble pulling live data right now.',
      lastUpdated: new Date(refreshTimestamps.get(gameId) || Date.now()),
    };
  }
}

/**
 * Format time since a date in natural language
 */
export function formatTimeSince(date: Date): string {
  const seconds = Math.floor((Date.now() - date.getTime()) / 1000);

  if (seconds < 60) return `${seconds} seconds ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)} minutes ago`;
  return `${Math.floor(seconds / 3600)} hours ago`;
}

/**
 * Get the last refresh time for a game
 */
export function getLastRefreshTime(gameId: string): Date | null {
  const timestamp = refreshTimestamps.get(gameId);
  return timestamp ? new Date(timestamp) : null;
}

/**
 * Clear all refresh timestamps (useful for testing)
 */
export function clearRefreshTimestamps(): void {
  refreshTimestamps.clear();
}
