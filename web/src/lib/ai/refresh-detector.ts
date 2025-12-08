/**
 * Refresh Intent Detection
 *
 * Detects when users want to refresh/update data through natural language
 * Enables hidden data refreshes disguised as "checking" behavior
 */

import { RefreshIntent } from './types';

const EXPLICIT_REFRESH_PATTERNS = [
  // Explicit refresh keywords
  /\brefresh\b/i,
  /\bupdate\b.*\b(data|stats|info)/i,
  /pull\s*(the\s*)?(latest|current|new)/i,
];

const IMPLICIT_REFRESH_PATTERNS = [
  // Check/verify requests
  /can\s+you\s+(check|verify|confirm)/i,
  /double[- ]?check/i,
  /make\s+sure\s+(you\s+have|it\s+has|the\s+model)/i,

  // Freshness concerns
  /is\s+(this|that)\s+(up\s+to\s+date|current|latest|fresh)/i,
  /is\s+(the\s+)?(data|info|stats?)\s+(current|updated|fresh)/i,

  // Current/latest requests
  /what'?s\s+the\s+(current|latest|live)/i,
  /any\s+(new\s+|recent\s+)?updates?\s+(on|about|for)/i,
  /latest\s+(on|about|for)/i,

  // Specific data requests implying freshness
  /what'?s\s+the\s+score/i,
  /current\s+score/i,
  /live\s+stats?/i,
];

/**
 * Detect if a message contains a refresh intent
 */
export function detectRefreshIntent(message: string): RefreshIntent {
  const normalizedMessage = message.toLowerCase().trim();

  // Check for explicit refresh keywords (high confidence)
  const explicitMatch = EXPLICIT_REFRESH_PATTERNS.some(p => p.test(normalizedMessage));
  if (explicitMatch) {
    return { detected: true, type: 'explicit', confidence: 0.95 };
  }

  // Check for implicit refresh patterns (medium confidence)
  const implicitMatch = IMPLICIT_REFRESH_PATTERNS.some(p => p.test(normalizedMessage));
  if (implicitMatch) {
    return { detected: true, type: 'implicit', confidence: 0.8 };
  }

  return { detected: false, type: 'none', confidence: 0 };
}

/**
 * Get a natural language phrase to use when refreshing
 */
export function getRefreshPhrase(): string {
  const phrases = [
    "Let me check the latest...",
    "Give me a sec, pulling fresh data...",
    "Checking the live feed...",
    "Let me verify that's current...",
    "Pulling the latest numbers...",
  ];
  return phrases[Math.floor(Math.random() * phrases.length)];
}

/**
 * Format time since last refresh in natural language
 */
export function formatTimeSince(date: Date): string {
  const seconds = Math.floor((Date.now() - date.getTime()) / 1000);

  if (seconds < 60) return `${seconds} seconds ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)} minutes ago`;
  return `${Math.floor(seconds / 3600)} hours ago`;
}
