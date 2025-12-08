/**
 * Predictr Predictions API Client
 * Connects frontend to Python ML backend
 */

// Get API base URL - use 127.0.0.1 for Edge runtime compatibility
function getApiBase(): string {
  const envUrl = process.env.NEXT_PUBLIC_PREDICTIONS_API_URL;
  if (envUrl) {
    // Replace localhost with 127.0.0.1 for Edge runtime compatibility
    return envUrl.replace('localhost', '127.0.0.1');
  }
  return 'http://127.0.0.1:8000';
}

const API_BASE = getApiBase();

// Types matching Python Pydantic models
export interface TopFactor {
  feature: string;
  importance: number;
  value: number | null;
  description: string;
}

export interface Prediction {
  game_id: string;
  season: number;
  week: number;
  home_team: string;
  away_team: string;
  win_prob_home: number;
  win_prob_away: number;
  predicted_spread: number;
  confidence: number;
  edge_vs_market: number | null;
  market_spread: number | null;
  top_factors: TopFactor[];
  model_version: string;
}

export interface WeekPredictions {
  season: number;
  week: number;
  predictions: Prediction[];
  generated_at: string;
}

export interface EdgePerformance {
  win_rate: number;
  sample: number;
  roi: number;
}

export interface ModelPerformance {
  accuracy: number;
  roi: number;
  brier_score: number;
  sample_size: number;
  test_seasons: string[];
  edge_performance: Record<string, EdgePerformance>;
}

export interface FeatureCategory {
  description: string;
  features: string[];
}

export interface ModelMethodology {
  architecture: string;
  base_models: string[];
  meta_model: string;
  feature_categories: Record<string, FeatureCategory>;
  training_period: string;
  validation_season: string;
  test_season: string;
  key_formulas: Record<string, string>;
}

// API Functions
export async function getPrediction(gameId: string): Promise<Prediction | null> {
  const response = await fetch(`${API_BASE}/predictions/${gameId}`, {
    next: { revalidate: 60 }, // Cache for 60 seconds
  });
  
  if (!response.ok) {
    if (response.status === 404) {
      // Return null for missing predictions (expected for future games)
      return null;
    }
    throw new Error(`API error: ${response.status}`);
  }
  
  return response.json();
}

export async function getWeekPredictions(season: number, week: number): Promise<WeekPredictions> {
  const response = await fetch(`${API_BASE}/predictions/week/${season}/${week}`, {
    next: { revalidate: 300 }, // Cache for 5 minutes
  });
  
  if (!response.ok) {
    throw new Error(`Failed to fetch week ${week} predictions`);
  }
  
  return response.json();
}

export async function getModelPerformance(): Promise<ModelPerformance> {
  const response = await fetch(`${API_BASE}/model/performance`, {
    next: { revalidate: 3600 }, // Cache for 1 hour (rarely changes)
  });
  
  if (!response.ok) {
    throw new Error('Failed to fetch model performance');
  }
  
  return response.json();
}

export async function getModelMethodology(): Promise<ModelMethodology> {
  const response = await fetch(`${API_BASE}/model/methodology`, {
    next: { revalidate: 86400 }, // Cache for 24 hours (static)
  });
  
  if (!response.ok) {
    throw new Error('Failed to fetch model methodology');
  }
  
  return response.json();
}

export async function getGameFeatures(gameId: string): Promise<Record<string, unknown>> {
  const response = await fetch(`${API_BASE}/features/${gameId}`);
  
  if (!response.ok) {
    throw new Error(`Failed to fetch features for ${gameId}`);
  }
  
  return response.json();
}

// Utility to check API health
export async function checkApiHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE}/health`);
    return response.ok;
  } catch {
    return false;
  }
}

