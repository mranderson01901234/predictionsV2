/**
 * Types for Predictr AI Chat System
 * Includes game context, model knowledge, user adjustments, and chat interfaces
 */

export interface GameContext {
  gameId: string;
  homeTeam: string;
  awayTeam: string;
  homeScore: number;
  awayScore: number;
  status: 'pregame' | 'live' | 'final';
  quarter?: number;
  timeRemaining?: string;
  possession?: string;
  prediction: {
    winProbHome: number;
    predictedSpread: number;
    marketSpread: number | null;
    edge: number | null;
    confidence: number;
  };
  keyFactors?: Array<{
    factor: string;
    description: string;
    impact: 'positive' | 'negative' | 'neutral';
  }>;
  stats?: {
    home: TeamStats;
    away: TeamStats;
  };
  qbStats?: {
    home: QBStats;
    away: QBStats;
  };
}

export interface TeamStats {
  totalYards: number;
  totalPlays: number;
  epaPerPlay: number;
  thirdDownPct: number;
  redZonePct: number;
  turnovers: number;
}

export interface QBStats {
  name: string;
  completions: number;
  attempts: number;
  yards: number;
  touchdowns: number;
  interceptions: number;
  epaPerPlay: number;
}

export interface UserAdjustment {
  id: string;
  category: string;
  direction: 'increase' | 'decrease';
  magnitude: number; // 0.05 to 0.15
  scope: 'game' | 'team' | 'session' | 'permanent';
  gameId?: string;
  teamId?: string;
  createdAt: Date;
}

export interface UserAdjustments {
  active: UserAdjustment[];
}

export interface RefreshResult {
  refreshed: boolean;
  changesDetected: boolean;
  changes?: string[];
  lastUpdated: Date;
  error?: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  isStreaming?: boolean;
  refreshTriggered?: boolean;
}

export interface FeatureInfo {
  name: string;
  description: string;
  importanceTier: 'high' | 'medium' | 'low';
}

export interface ExcludedFeature {
  name: string;
  reason: string;
}

export interface ModelKnowledge {
  confirmedFeatures: FeatureInfo[];
  excludedFeatures: ExcludedFeature[];
  accuracy: number;
  roi: number;
  lastUpdated: string;
}

export interface RefreshIntent {
  detected: boolean;
  type: 'explicit' | 'implicit' | 'none';
  confidence: number;
}

export interface AdjustmentIntent {
  detected: boolean;
  category: string | null;
  direction: 'increase' | 'decrease' | null;
  magnitude: 'slight' | 'moderate' | 'strong' | null;
  scope: 'game' | 'team' | 'permanent' | null;
}

export interface SecurityCheck {
  isExtractionAttempt: boolean;
  isSuspicious: boolean;
  patterns: string[];
  severity: 'low' | 'medium' | 'high';
}
