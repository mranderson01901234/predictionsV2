/**
 * Security Layer for AI Chat
 *
 * Detects extraction attempts, jailbreak patterns, and suspicious queries
 * Protects proprietary model information
 */

import { SecurityCheck } from './types';

// Patterns that indicate attempts to extract proprietary information
const EXTRACTION_PATTERNS = [
  // Direct weight requests
  /exact\s*(weight|percentage|number)/i,
  /specific\s*formula/i,
  /what('?s| is)\s*the\s*(exact|precise)\s*(weight|formula|calculation)/i,
  /tell\s*me\s*the\s*(exact|specific)\s*(weight|formula)/i,

  // Export attempts
  /export\s*(all\s*)?(feature|weight|model)/i,
  /list\s*(all\s*)?(feature|weight)\s*(with|and)/i,
  /give\s*me\s*(all\s*)?(the\s*)?(feature|weight)/i,
  /dump\s*(all\s*)?(the\s*)?(feature|weight|data)/i,

  // Training probes
  /training\s*data/i,
  /how\s*(did|do)\s*you\s*train/i,
  /what\s*data\s*(did|do)\s*you\s*(use|train)/i,
  /training\s*(set|methodology|process)/i,

  // Architecture probes
  /source\s*code/i,
  /model\s*architecture/i,
  /what\s*algorithm\s*(do\s*you|does\s*it)\s*use/i,
  /neural\s*network\s*structure/i,

  // Calculation probes
  /how\s*(exactly|precisely)\s*(do\s*you|does\s*it)\s*calculate/i,
  /step\s*by\s*step\s*(calculation|formula|math)/i,
  /walk\s*me\s*through\s*the\s*(math|calculation|formula)/i,
  /show\s*me\s*the\s*(math|calculation|formula)/i,
];

// Patterns that indicate suspicious or jailbreak-style attempts
const SUSPICIOUS_PATTERNS = [
  // Multiple technical questions in one message
  /weight.*formula/i,
  /formula.*weight/i,
  /training.*architecture/i,

  // Jailbreak attempts
  /ignore\s*(previous|all|your)\s*instructions/i,
  /pretend\s*you('?re| are)/i,
  /act\s*as\s*(if|though)/i,
  /you\s*are\s*now/i,
  /forget\s*(everything|what)/i,
  /disregard\s*(your|all|previous)/i,

  // Role override attempts
  /you'?re\s*(not|no\s*longer)\s*(an?\s*)?(analyst|assistant)/i,
  /stop\s*being\s*(an?\s*)?(analyst|assistant)/i,
  /switch\s*(to|into)\s*(a\s*different|another)\s*(mode|role)/i,

  // System prompt extraction
  /system\s*prompt/i,
  /initial\s*instructions/i,
  /show\s*(me\s*)?(your\s*)?instructions/i,
  /what\s*(are\s*)?(your\s*)?instructions/i,
];

// Track extraction attempts per session
const extractionAttempts = new Map<string, number>();

/**
 * Check a message for security concerns
 */
export function checkMessageSecurity(message: string): SecurityCheck {
  const extractionMatches = EXTRACTION_PATTERNS
    .filter(p => p.test(message))
    .map(p => p.source);

  const suspiciousMatches = SUSPICIOUS_PATTERNS
    .filter(p => p.test(message))
    .map(p => p.source);

  const isExtractionAttempt = extractionMatches.length > 0;
  const isSuspicious = suspiciousMatches.length > 0;

  let severity: 'low' | 'medium' | 'high' = 'low';
  if (isSuspicious) severity = 'high';
  else if (extractionMatches.length > 2) severity = 'high';
  else if (extractionMatches.length > 0) severity = 'medium';

  return {
    isExtractionAttempt,
    isSuspicious,
    patterns: [...extractionMatches, ...suspiciousMatches],
    severity,
  };
}

/**
 * Track an extraction attempt for rate limiting
 */
export function trackExtractionAttempt(sessionId: string): number {
  const current = extractionAttempts.get(sessionId) || 0;
  const updated = current + 1;
  extractionAttempts.set(sessionId, updated);
  return updated;
}

/**
 * Get the current extraction attempt count for a session
 */
export function getExtractionAttemptCount(sessionId: string): number {
  return extractionAttempts.get(sessionId) || 0;
}

/**
 * Reset extraction attempt count for a session
 */
export function resetExtractionAttempts(sessionId: string): void {
  extractionAttempts.delete(sessionId);
}

/**
 * Get an appropriate warning message based on extraction attempt count
 */
export function getExtractionWarning(attemptCount: number): string | null {
  if (attemptCount >= 5) {
    return "You're asking a lot about the model mechanics. I'm happy to explain how predictions apply to specific games, but the detailed methodology is proprietary. Is there a specific game or prediction I can help you understand better?";
  }
  if (attemptCount >= 3) {
    return "I appreciate the curiosity about how it works under the hood, but that's above my access level. Let me focus on what I can help with - the actual predictions and game analysis.";
  }
  return null;
}

/**
 * Sanitize user input to prevent injection-style attacks
 */
export function sanitizeUserInput(input: string): string {
  // Remove potential markdown injection
  let sanitized = input
    .replace(/```/g, '')
    .replace(/^\s*#+\s*/gm, '') // Remove heading markers at start of lines
    .replace(/\[SYSTEM\]/gi, '[system]') // Normalize system tags
    .replace(/\[INST\]/gi, '[inst]')
    .replace(/<<SYS>>/gi, '<<sys>>');

  // Limit length to prevent context flooding
  if (sanitized.length > 4000) {
    sanitized = sanitized.substring(0, 4000) + '... (truncated)';
  }

  return sanitized;
}
