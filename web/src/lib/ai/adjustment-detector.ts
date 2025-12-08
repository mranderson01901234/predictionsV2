/**
 * Adjustment Intent Detection
 *
 * Detects when users want to adjust model weights through natural language
 * Enables natural language model tuning with guardrails
 */

import { AdjustmentIntent } from './types';

// Category detection patterns
const CATEGORY_PATTERNS: Record<string, RegExp[]> = {
  qb_performance: [
    /\b(qb|quarterback)\b/i,
    /\b(passing|pass)\s*(game|efficiency|performance)?/i,
    /\b(lawrence|richardson|mahomes|allen|burrow|herbert|goff)\b/i, // Common QB names
    /\bqbr\b/i,
    /\bpassing\s*epa\b/i,
  ],
  rushing: [
    /\b(rush|rushing|run|ground)\s*(game|attack|efficiency)?/i,
    /\b(rb|running\s*back)/i,
    /\brush\s*epa\b/i,
  ],
  defense: [
    /\b(defense|defensive|d-line|secondary)/i,
    /\bdefend/i,
    /\bdef\s*epa\b/i,
  ],
  recent_form: [
    /\b(hot|cold)\s*(streak|run)/i,
    /\brecent\s*(form|games|performance)/i,
    /\bmomentum\b/i,
    /\btrending/i,
    /\brolling\b/i,
  ],
  turnovers: [
    /\bturnover/i,
    /\b(fumble|interception|pick)/i,
    /\bgiveaway/i,
    /\btakeaway/i,
  ],
  home_field: [
    /\bhome\s*(field|crowd|advantage)/i,
    /\bdome\b/i,
    /\bcrowd\s*(noise|factor)/i,
  ],
  third_down: [
    /\b3rd\s*down/i,
    /\bthird\s*down/i,
    /\bconversion/i,
  ],
  red_zone: [
    /\bred\s*zone/i,
    /\bscoring\s*efficiency/i,
  ],
};

// Direction detection patterns
const INCREASE_PATTERNS = [
  /\b(increase|bump|raise|trust|weight|favor|more|higher)\b/i,
  /\b(is|looks?)\s*(elite|great|amazing|legit|underrated)/i,
  /\bunderweight/i,
  /\bnot\s+enough\s+credit/i,
  /\btoo\s+low\b/i,
];

const DECREASE_PATTERNS = [
  /\b(decrease|lower|reduce|less|ignore|discount|fade)\b/i,
  /\bdon'?t\s*trust/i,
  /\boverrat/i,
  /\boverweight/i,
  /\btoo\s+(high|much)\b/i,
  /\bnot\s+(as\s+)?(good|important)/i,
];

// Magnitude detection patterns
const MAGNITUDE_PATTERNS = {
  slight: [/\b(slightly|a\s*bit|a\s*little|nudge|tiny)/i],
  moderate: [/\b(somewhat|moderately|noticeably)/i],
  strong: [/\b(significantly|heavily|a\s*lot|really|crank|max|way\s*more)/i],
};

// Reset patterns
const RESET_PATTERNS = [
  /\breset\b/i,
  /\b(go\s*)?back\s*to\s*(default|normal|original)/i,
  /\bclear\s*(all\s*)?(adjustment|tweak|change)/i,
  /\bundo\b/i,
  /\bremove\s*(all\s*)?(adjustment|tweak)/i,
];

/**
 * Detect if a message contains an adjustment intent
 */
export function detectAdjustmentIntent(message: string): AdjustmentIntent {
  // Check for reset first
  if (RESET_PATTERNS.some(p => p.test(message))) {
    return {
      detected: true,
      category: 'reset',
      direction: null,
      magnitude: null,
      scope: null,
    };
  }

  // Detect category
  let category: string | null = null;
  for (const [cat, patterns] of Object.entries(CATEGORY_PATTERNS)) {
    if (patterns.some(p => p.test(message))) {
      category = cat;
      break;
    }
  }

  if (!category) {
    return { detected: false, category: null, direction: null, magnitude: null, scope: null };
  }

  // Detect direction
  let direction: 'increase' | 'decrease' | null = null;
  if (INCREASE_PATTERNS.some(p => p.test(message))) {
    direction = 'increase';
  } else if (DECREASE_PATTERNS.some(p => p.test(message))) {
    direction = 'decrease';
  }

  // Detect magnitude (default to moderate)
  let magnitude: 'slight' | 'moderate' | 'strong' | null = 'moderate';
  for (const [mag, patterns] of Object.entries(MAGNITUDE_PATTERNS)) {
    if (patterns.some(p => p.test(message))) {
      magnitude = mag as 'slight' | 'moderate' | 'strong';
      break;
    }
  }

  // Detect scope (default to 'game')
  let scope: 'game' | 'team' | 'permanent' | null = 'game';
  if (/\b(permanent|always|from\s*now\s*on|going\s*forward)/i.test(message)) {
    scope = 'permanent';
  } else if (/\b(for\s*)?(this\s*)?team\b/i.test(message)) {
    scope = 'team';
  }

  return {
    detected: direction !== null,
    category,
    direction,
    magnitude,
    scope,
  };
}

/**
 * Convert magnitude string to numeric value
 */
export function magnitudeToValue(magnitude: 'slight' | 'moderate' | 'strong' | null): number {
  switch (magnitude) {
    case 'slight': return 0.05;
    case 'moderate': return 0.10;
    case 'strong': return 0.15;
    default: return 0.10;
  }
}

/**
 * Get human-readable description of an adjustment
 */
export function describeAdjustment(intent: AdjustmentIntent): string {
  if (intent.category === 'reset') {
    return 'Reset all adjustments to default weights';
  }

  if (!intent.detected || !intent.category || !intent.direction) {
    return '';
  }

  const categoryLabels: Record<string, string> = {
    qb_performance: 'QB performance',
    rushing: 'rushing game',
    defense: 'defensive metrics',
    recent_form: 'recent form',
    turnovers: 'turnover impact',
    home_field: 'home field advantage',
    third_down: 'third down efficiency',
    red_zone: 'red zone scoring',
  };

  const category = categoryLabels[intent.category] || intent.category;
  const direction = intent.direction === 'increase' ? 'increase' : 'decrease';
  const magnitude = intent.magnitude || 'moderate';

  return `${magnitude.charAt(0).toUpperCase() + magnitude.slice(1)} ${direction} to ${category} weight`;
}
