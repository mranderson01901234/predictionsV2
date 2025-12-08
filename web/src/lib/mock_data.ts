import { getQBHeadshotSync, preloadQBHeadshots } from './services/qb-headshots';

export interface Game {
    game_id: string;
    season: number;
    week: number;
    date: string;
    home_team: string;
    away_team: string;
    home_score: number;
    away_score: number;
    status: 'Scheduled' | 'Live' | 'Final';
    quarter?: number;
    time_remaining?: string;
    possession?: 'home' | 'away';
    home_record?: string;
    away_record?: string;
    home_logo?: string;
    away_logo?: string;
}

export interface MarketSnapshot {
    game_id: string;
    bookmaker: string;
    spread_home: number;
    total: number;
    spread_home_open?: number;
    total_open?: number;
}

export interface Prediction {
    game_id: string;
    win_prob_home: number;
    predicted_spread: number;
    predicted_total: number;
    confidence_score: number; // 0-100
    edge_spread: number; // Model - Market
    edge_total: number;
}

// Mock Data
export const MOCK_GAMES: Game[] = [
    {
        game_id: "nfl_2025_14_IND_JAX",
        season: 2025,
        week: 14,
        date: "2025-12-07T13:00:00Z",
        home_team: "JAX",
        away_team: "IND",
        home_score: 24,
        away_score: 17,
        status: "Live",
        quarter: 3,
        time_remaining: "08:45",
        possession: "home",
        home_record: "2-10",
        away_record: "9-3",
        home_logo: "https://a.espncdn.com/i/teamlogos/nfl/500/jax.png",
        away_logo: "https://a.espncdn.com/i/teamlogos/nfl/500/ind.png"
    },
    {
        game_id: "nfl_2025_14_BUF_DET",
        season: 2025,
        week: 14,
        date: "2025-12-07T16:25:00Z",
        home_team: "DET",
        away_team: "BUF",
        home_score: 0,
        away_score: 0,
        status: "Scheduled",
        home_record: "7-5",
        away_record: "8-4",
        home_logo: "https://a.espncdn.com/i/teamlogos/nfl/500/det.png",
        away_logo: "https://a.espncdn.com/i/teamlogos/nfl/500/buf.png"
    },
    {
        game_id: "nfl_2025_14_KC_LAC",
        season: 2025,
        week: 14,
        date: "2025-12-07T20:20:00Z",
        home_team: "LAC",
        away_team: "KC",
        home_score: 0,
        away_score: 0,
        status: "Scheduled",
        home_record: "6-6",
        away_record: "11-1",
        home_logo: "https://a.espncdn.com/i/teamlogos/nfl/500/lac.png",
        away_logo: "https://a.espncdn.com/i/teamlogos/nfl/500/kc.png"
    }
];

export const MOCK_MARKETS: Record<string, MarketSnapshot> = {
    "nfl_2025_14_IND_JAX": {
        game_id: "nfl_2025_14_IND_JAX",
        bookmaker: "Consensus",
        spread_home: -3.5,
        total: 46.5
    },
    "nfl_2025_14_BUF_DET": {
        game_id: "nfl_2025_14_BUF_DET",
        bookmaker: "Consensus",
        spread_home: -2.0,
        total: 54.0
    },
    "nfl_2025_14_KC_LAC": {
        game_id: "nfl_2025_14_KC_LAC",
        bookmaker: "Consensus",
        spread_home: 4.5,
        total: 48.5
    }
};

export const MOCK_PREDICTIONS: Record<string, Prediction> = {
    "nfl_2025_14_IND_JAX": {
        game_id: "nfl_2025_14_IND_JAX",
        win_prob_home: 0.65,
        predicted_spread: -5.2,
        predicted_total: 45.0,
        confidence_score: 85,
        edge_spread: 1.7, // Model says -5.2, Market says -3.5. Edge is 1.7 points of value on Home? Or Away?
        // If market is -3.5 (Home favored by 3.5) and model is -5.2 (Home favored by 5.2),
        // Model thinks Home is stronger than market. Bet Home.
        edge_total: -1.5
    },
    "nfl_2025_14_BUF_DET": {
        game_id: "nfl_2025_14_BUF_DET",
        win_prob_home: 0.52,
        predicted_spread: -0.5,
        predicted_total: 56.5,
        confidence_score: 60,
        edge_spread: -1.5, // Market -2.0, Model -0.5. Model thinks Home is weaker. Bet Away.
        edge_total: 2.5
    },
    "nfl_2025_14_KC_LAC": {
        game_id: "nfl_2025_14_KC_LAC",
        win_prob_home: 0.30,
        predicted_spread: 7.5,
        predicted_total: 47.0,
        confidence_score: 92,
        edge_spread: -3.0,
        edge_total: -1.5
    }
};

export async function getGames(): Promise<Game[]> {
    // Simulate network delay
    await new Promise(resolve => setTimeout(resolve, 500));
    return MOCK_GAMES;
}

export interface Quarterback {
    name: string;
    team: string;
    headshot_url: string;
    completions: number;
    attempts: number;
    yards: number;
    tds: number;
    ints: number;
    epa: number;
    qbr: number;
}

/**
 * Enrich QB data with headshot URL from the headshots database
 * Always prefers the lookup from JSON file over hardcoded values
 */
function enrichQBWithHeadshot(qb: Omit<Quarterback, 'headshot_url'> & { headshot_url?: string }): Quarterback {
    // Always try to get headshot from JSON first, fallback to existing if not found
    const lookupHeadshot = getQBHeadshotSync(qb.name);
    const headshot = lookupHeadshot || qb.headshot_url || '';
    return {
        ...qb,
        headshot_url: headshot
    };
}

export interface TeamStats {
    total_yards: number;
    plays: number;
    yards_per_play: number;
    epa_per_play: number;
    passing_yards: number;
    rushing_yards: number;
    turnovers: number;
    first_downs: number;
    third_down_conv: string; // "4/12"
    possession: string; // "30:15"
    // Detailed stats
    comp_att?: string;
    yards_per_pass?: number;
    epa_per_pass?: number;
    rush_attempts?: number;
    yards_per_rush?: number;
    epa_per_rush?: number;
    passing_first_downs?: number;
    rushing_first_downs?: number;
    penalty_first_downs?: number;
    red_zone_conv?: string;
    sacks?: number;
    sack_yards?: number;
    penalties?: number;
    penalty_yards?: number;
}

export interface GameDetail extends Game {
    home_qb: Quarterback;
    away_qb: Quarterback;
    home_stats: TeamStats;
    away_stats: TeamStats;
    scoring_summary: {
        home: number[]; // [Q1, Q2, Q3, Q4]
        away: number[];
    };
    win_probability: { time: string; home_prob: number }[];
    market?: MarketSnapshot;
    prediction?: Prediction;
}

export const MOCK_GAME_DETAILS: Record<string, GameDetail> = {
    "nfl_2025_14_IND_JAX": {
        ...MOCK_GAMES[0],
        home_qb: {
            name: "Trevor Lawrence",
            team: "JAX",
            headshot_url: "https://static.www.nfl.com/image/private/f_auto,q_auto/league/g44jngk8039i97a4h86t",
            completions: 22,
            attempts: 34,
            yards: 285,
            tds: 2,
            ints: 0,
            epa: 8.4,
            qbr: 72.5
        },
        away_qb: {
            name: "Anthony Richardson",
            team: "IND",
            headshot_url: "https://static.www.nfl.com/image/private/f_auto,q_auto/league/jfmq6q8g4g8g4g8g4g8g",
            completions: 14,
            attempts: 28,
            yards: 195,
            tds: 1,
            ints: 1,
            epa: -2.1,
            qbr: 45.2
        },
        home_stats: {
            total_yards: 385,
            plays: 64,
            yards_per_play: 6.0,
            epa_per_play: 0.15,
            passing_yards: 285,
            rushing_yards: 100,
            turnovers: 0,
            first_downs: 22,
            third_down_conv: "6/13",
            possession: "32:10",
            comp_att: "22/34",
            yards_per_pass: 8.4,
            epa_per_pass: 0.25,
            rush_attempts: 30,
            yards_per_rush: 3.3,
            epa_per_rush: -0.05,
            passing_first_downs: 14,
            rushing_first_downs: 6,
            penalty_first_downs: 2,
            red_zone_conv: "2/3",
            sacks: 1,
            sack_yards: 8,
            penalties: 4,
            penalty_yards: 35
        },
        away_stats: {
            total_yards: 290,
            plays: 58,
            yards_per_play: 5.0,
            epa_per_play: -0.05,
            passing_yards: 195,
            rushing_yards: 95,
            turnovers: 2,
            first_downs: 16,
            third_down_conv: "4/12",
            possession: "27:50",
            comp_att: "14/28",
            yards_per_pass: 7.0,
            epa_per_pass: -0.15,
            rush_attempts: 25,
            yards_per_rush: 3.8,
            epa_per_rush: -0.10,
            passing_first_downs: 9,
            rushing_first_downs: 5,
            penalty_first_downs: 2,
            red_zone_conv: "1/2",
            sacks: 3,
            sack_yards: 22,
            penalties: 6,
            penalty_yards: 50
        },
        scoring_summary: {
            home: [7, 10, 0, 7],
            away: [3, 0, 7, 7]
        },
        win_probability: [
            { time: "Q1 15:00", home_prob: 0.55 },
            { time: "Q1 08:00", home_prob: 0.62 },
            { time: "Q2 15:00", home_prob: 0.68 },
            { time: "Q2 00:00", home_prob: 0.75 },
            { time: "Q3 08:00", home_prob: 0.65 },
            { time: "Q4 15:00", home_prob: 0.72 },
            { time: "Q4 05:00", home_prob: 0.85 }
        ]
    },
    // Add defaults for others to prevent crashes
    "nfl_2025_14_BUF_DET": {
        ...MOCK_GAMES[1],
        home_qb: { name: "Jared Goff", team: "DET", headshot_url: "", completions: 0, attempts: 0, yards: 0, tds: 0, ints: 0, epa: 0, qbr: 0 },
        away_qb: { name: "Josh Allen", team: "BUF", headshot_url: "", completions: 0, attempts: 0, yards: 0, tds: 0, ints: 0, epa: 0, qbr: 0 },
        home_stats: { total_yards: 0, plays: 0, yards_per_play: 0, epa_per_play: 0, passing_yards: 0, rushing_yards: 0, turnovers: 0, first_downs: 0, third_down_conv: "0/0", possession: "00:00" },
        away_stats: { total_yards: 0, plays: 0, yards_per_play: 0, epa_per_play: 0, passing_yards: 0, rushing_yards: 0, turnovers: 0, first_downs: 0, third_down_conv: "0/0", possession: "00:00" },
        scoring_summary: { home: [0, 0, 0, 0], away: [0, 0, 0, 0] },
        win_probability: [{ time: "Pre", home_prob: 0.5 }]
    },
    "nfl_2025_14_KC_LAC": {
        ...MOCK_GAMES[2],
        home_qb: { name: "Justin Herbert", team: "LAC", headshot_url: "", completions: 0, attempts: 0, yards: 0, tds: 0, ints: 0, epa: 0, qbr: 0 },
        away_qb: { name: "Patrick Mahomes", team: "KC", headshot_url: "", completions: 0, attempts: 0, yards: 0, tds: 0, ints: 0, epa: 0, qbr: 0 },
        home_stats: { total_yards: 0, plays: 0, yards_per_play: 0, epa_per_play: 0, passing_yards: 0, rushing_yards: 0, turnovers: 0, first_downs: 0, third_down_conv: "0/0", possession: "00:00" },
        away_stats: { total_yards: 0, plays: 0, yards_per_play: 0, epa_per_play: 0, passing_yards: 0, rushing_yards: 0, turnovers: 0, first_downs: 0, third_down_conv: "0/0", possession: "00:00" },
        scoring_summary: { home: [0, 0, 0, 0], away: [0, 0, 0, 0] },
        win_probability: [{ time: "Pre", home_prob: 0.5 }]
    }
};

export async function getGameDetails(gameId: string) {
    await new Promise(resolve => setTimeout(resolve, 300));
    const game = MOCK_GAMES.find(g => g.game_id === gameId);
    const market = MOCK_MARKETS[gameId];
    const prediction = MOCK_PREDICTIONS[gameId];

    return { game, market, prediction };
}

export async function getFullGameDetails(gameId: string) {
    await new Promise(resolve => setTimeout(resolve, 300));
    // Preload QB headshots if not already loaded
    await preloadQBHeadshots();
    
    const details = MOCK_GAME_DETAILS[gameId];
    if (!details) {
        // Return null if game not found
        return null;
    }
    // Enrich QB data with headshots
    const enrichedDetails = {
        ...details,
        home_qb: enrichQBWithHeadshot(details.home_qb),
        away_qb: enrichQBWithHeadshot(details.away_qb),
    };
    const market = MOCK_MARKETS[gameId];
    const prediction = MOCK_PREDICTIONS[gameId];
    return { ...enrichedDetails, market, prediction };
}

export async function getAllGameDetails() {
    await new Promise(resolve => setTimeout(resolve, 300));
    // Preload QB headshots if not already loaded
    await preloadQBHeadshots();
    
    const allDetails: Record<string, GameDetail> = {};
    for (const gameId of Object.keys(MOCK_GAME_DETAILS)) {
        const details = MOCK_GAME_DETAILS[gameId];
        // Enrich QB data with headshots
        const enrichedDetails = {
            ...details,
            home_qb: enrichQBWithHeadshot(details.home_qb),
            away_qb: enrichQBWithHeadshot(details.away_qb),
        };
        const market = MOCK_MARKETS[gameId];
        const prediction = MOCK_PREDICTIONS[gameId];
        allDetails[gameId] = { ...enrichedDetails, market, prediction };
    }
    return allDetails;
}
