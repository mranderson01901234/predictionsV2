/**
 * Model Knowledge Configuration
 * Defines what the AI analyst "knows" about the prediction model
 * Includes confirmed features, excluded features, and verified performance
 */

import { ModelKnowledge, FeatureInfo, ExcludedFeature } from './types';

export const MODEL_KNOWLEDGE: ModelKnowledge = {
  confirmedFeatures: [
    // High importance
    {
      name: 'Team Offensive EPA',
      description: 'Expected points added per offensive play',
      importanceTier: 'high',
    },
    {
      name: 'Team Defensive EPA',
      description: 'Expected points allowed per defensive play',
      importanceTier: 'high',
    },
    {
      name: 'QB EPA per Dropback',
      description: 'Quarterback efficiency on passing plays',
      importanceTier: 'high',
    },
    {
      name: 'Rolling Team Form',
      description: 'Win rate and point differential over recent games',
      importanceTier: 'high',
    },

    // Medium importance
    {
      name: 'QB CPOE',
      description: 'Completion percentage over expected',
      importanceTier: 'medium',
    },
    {
      name: 'Turnover Differential',
      description: 'Net turnovers (takeaways minus giveaways)',
      importanceTier: 'medium',
    },
    {
      name: 'Third Down Efficiency',
      description: '3rd down conversion rate (offense and defense)',
      importanceTier: 'medium',
    },
    {
      name: 'Red Zone Efficiency',
      description: 'Scoring percentage in red zone',
      importanceTier: 'medium',
    },
    {
      name: 'Rushing EPA',
      description: 'Expected points added on rushing plays',
      importanceTier: 'medium',
    },

    // Low importance
    {
      name: 'Sack Rate',
      description: 'Sacks allowed/generated per pass attempt',
      importanceTier: 'low',
    },
    {
      name: 'Home Field',
      description: 'Home team adjustment factor',
      importanceTier: 'low',
    },
  ],

  excludedFeatures: [
    {
      name: 'Weather (explicit)',
      reason: 'Not a direct feature, but affects stats that are included',
    },
    {
      name: 'Travel distance',
      reason: 'Not currently in the model',
    },
    {
      name: 'Revenge game / narratives',
      reason: 'Qualitative factors not quantified',
    },
    {
      name: 'Playoff implications',
      reason: 'Motivation factors not directly modeled',
    },
    {
      name: 'Primetime adjustments',
      reason: 'No specific game-time adjustment',
    },
    {
      name: 'Referee tendencies',
      reason: 'Not included in current feature set',
    },
  ],

  accuracy: 67,
  roi: 28,
  lastUpdated: '2025-12-08',
};

/**
 * Check if a feature name matches any confirmed feature
 */
export function isConfirmedFeature(featureName: string): FeatureInfo | null {
  const normalized = featureName.toLowerCase();
  return MODEL_KNOWLEDGE.confirmedFeatures.find(f =>
    f.name.toLowerCase().includes(normalized) ||
    normalized.includes(f.name.toLowerCase())
  ) || null;
}

/**
 * Check if a feature is explicitly excluded
 */
export function isExcludedFeature(featureName: string): ExcludedFeature | null {
  const normalized = featureName.toLowerCase();
  return MODEL_KNOWLEDGE.excludedFeatures.find(f =>
    f.name.toLowerCase().includes(normalized) ||
    normalized.includes(f.name.toLowerCase())
  ) || null;
}

/**
 * Get features by importance tier
 */
export function getFeaturesByTier(tier: 'high' | 'medium' | 'low'): FeatureInfo[] {
  return MODEL_KNOWLEDGE.confirmedFeatures.filter(f => f.importanceTier === tier);
}
