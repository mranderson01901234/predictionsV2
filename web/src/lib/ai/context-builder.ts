/**
 * Game Context Builder for AI Chat
 *
 * Formats game data into a structured context string for injection into LLM prompts.
 */

import { GameDetail, Prediction as MockPrediction } from "@/lib/mock_data";
import { Prediction as ApiPrediction } from "@/lib/api/predictions";

/**
 * Game State Interpreter
 * Calculates contextual labels for game situation awareness
 */
interface GameStateAnalysis {
  scoreDifferential: number;
  leadingTeam: string | null;
  possessionsNeeded: number; // How many scores needed to tie/take lead
  minutesRemaining: number;
  gamePhase: 'early' | 'mid' | 'late' | 'critical' | 'final';
  gameCloseness: 'coin_flip' | 'competitive' | 'comfortable' | 'commanding' | 'blowout' | 'decided';
  situationSummary: string;
  winProbInterpretation: string;
}

function parseTimeRemaining(timeStr: string | undefined): { minutes: number; seconds: number } {
  if (!timeStr) return { minutes: 0, seconds: 0 };
  const parts = timeStr.split(':');
  if (parts.length === 2) {
    return { minutes: parseInt(parts[0]) || 0, seconds: parseInt(parts[1]) || 0 };
  }
  return { minutes: 0, seconds: 0 };
}

function calculateMinutesRemaining(quarter: number | undefined, timeRemaining: string | undefined): number {
  if (!quarter) return 60; // Pregame, full game ahead

  const { minutes, seconds } = parseTimeRemaining(timeRemaining);
  const timeInQuarter = minutes + (seconds / 60);

  // NFL quarters are 15 minutes each
  const quartersRemaining = 4 - quarter;
  return (quartersRemaining * 15) + timeInQuarter;
}

function analyzeGameState(
  homeTeam: string,
  awayTeam: string,
  homeScore: number,
  awayScore: number,
  quarter: number | undefined,
  timeRemaining: string | undefined,
  winProbHome: number,
  status: string
): GameStateAnalysis {
  const scoreDiff = homeScore - awayScore;
  const absDiff = Math.abs(scoreDiff);
  const leadingTeam = scoreDiff > 0 ? homeTeam : scoreDiff < 0 ? awayTeam : null;
  const trailingTeam = scoreDiff > 0 ? awayTeam : scoreDiff < 0 ? homeTeam : null;

  // Calculate possessions needed (1 possession = up to 8 points with 2pt conversion)
  const possessionsNeeded = Math.ceil(absDiff / 8);

  // Calculate time remaining
  const minutesRemaining = status === 'Final' ? 0 : calculateMinutesRemaining(quarter, timeRemaining);

  // Determine game phase
  let gamePhase: GameStateAnalysis['gamePhase'];
  if (status === 'Final') {
    gamePhase = 'final';
  } else if (minutesRemaining > 45) {
    gamePhase = 'early'; // Q1 or early Q2
  } else if (minutesRemaining > 20) {
    gamePhase = 'mid'; // Late Q2 or Q3
  } else if (minutesRemaining > 5) {
    gamePhase = 'late'; // Q4 with time
  } else {
    gamePhase = 'critical'; // Under 5 minutes, every play matters
  }

  // Determine game closeness based on score AND time
  let gameCloseness: GameStateAnalysis['gameCloseness'];

  if (status === 'Final') {
    gameCloseness = 'decided';
  } else if (absDiff === 0) {
    gameCloseness = 'coin_flip';
  } else if (possessionsNeeded === 1) {
    // 1-8 point game (one score)
    if (minutesRemaining > 30) {
      gameCloseness = 'coin_flip'; // Plenty of time, essentially even
    } else if (minutesRemaining > 10) {
      gameCloseness = 'competitive'; // Close but leader has edge
    } else if (minutesRemaining > 2) {
      gameCloseness = 'comfortable'; // Leader in good shape but not over
    } else {
      gameCloseness = leadingTeam && winProbHome > 0.75 ? 'commanding' : 'competitive';
    }
  } else if (possessionsNeeded === 2) {
    // 9-16 point game (two scores)
    if (minutesRemaining > 20) {
      gameCloseness = 'competitive';
    } else if (minutesRemaining > 8) {
      gameCloseness = 'comfortable';
    } else {
      gameCloseness = 'commanding';
    }
  } else if (possessionsNeeded === 3) {
    // 17-24 point game (three scores)
    if (minutesRemaining > 25) {
      gameCloseness = 'comfortable';
    } else {
      gameCloseness = 'blowout';
    }
  } else {
    // 25+ point game
    gameCloseness = 'blowout';
  }

  // Generate situation summary
  let situationSummary: string;
  const timeDesc = minutesRemaining > 30 ? `${Math.round(minutesRemaining)} minutes (over half the game)` :
                   minutesRemaining > 15 ? `${Math.round(minutesRemaining)} minutes (roughly one quarter)` :
                   minutesRemaining > 5 ? `${Math.round(minutesRemaining)} minutes` :
                   minutesRemaining > 2 ? `under 5 minutes` :
                   `under 2 minutes (crunch time)`;

  if (status === 'Final') {
    situationSummary = `Game is over. Final score.`;
  } else if (absDiff === 0) {
    situationSummary = `Tied game with ${timeDesc} remaining. True toss-up.`;
  } else if (possessionsNeeded === 1) {
    situationSummary = `${leadingTeam} leads by ${absDiff} (ONE score). ${trailingTeam} needs just one TD to tie/take lead. With ${timeDesc} remaining, this is ${gameCloseness === 'coin_flip' ? 'essentially a coin flip' : gameCloseness === 'competitive' ? 'very much still competitive' : 'advantage ' + leadingTeam + ' but not decided'}.`;
  } else if (possessionsNeeded === 2) {
    situationSummary = `${leadingTeam} leads by ${absDiff} (TWO scores). ${trailingTeam} needs multiple scoring drives. With ${timeDesc} remaining, ${gameCloseness === 'competitive' ? 'comeback is definitely possible' : gameCloseness === 'comfortable' ? leadingTeam + ' is in control but game not over' : 'this is getting difficult for ' + trailingTeam}.`;
  } else {
    situationSummary = `${leadingTeam} leads by ${absDiff} (${possessionsNeeded} scores). With ${timeDesc} remaining, this is ${gameCloseness === 'blowout' ? 'effectively decided - would require historic comeback' : 'a significant deficit to overcome'}.`;
  }

  // Interpret win probability
  let winProbInterpretation: string;
  const winProb = Math.max(winProbHome, 1 - winProbHome);
  const favored = winProbHome > 0.5 ? homeTeam : awayTeam;

  if (winProb >= 0.95) {
    winProbInterpretation = `${favored} has 95%+ win probability - game is essentially decided barring miracle.`;
  } else if (winProb >= 0.85) {
    winProbInterpretation = `${favored} heavily favored (${(winProb * 100).toFixed(0)}%) - would need significant collapse to lose.`;
  } else if (winProb >= 0.75) {
    winProbInterpretation = `${favored} solidly favored (${(winProb * 100).toFixed(0)}%) - clear advantage but not locked in.`;
  } else if (winProb >= 0.65) {
    winProbInterpretation = `${favored} has edge (${(winProb * 100).toFixed(0)}%) - favored but very much a game. Outcome uncertain.`;
  } else if (winProb >= 0.55) {
    winProbInterpretation = `Slight edge to ${favored} (${(winProb * 100).toFixed(0)}%) - this is essentially a toss-up. Either team can win.`;
  } else {
    winProbInterpretation = `True coin flip (${(winProb * 100).toFixed(0)}%) - no meaningful favorite.`;
  }

  return {
    scoreDifferential: scoreDiff,
    leadingTeam,
    possessionsNeeded,
    minutesRemaining,
    gamePhase,
    gameCloseness,
    situationSummary,
    winProbInterpretation
  };
}

export interface UserBet {
  team: string;
  spread: number;
  amount?: number;
}

// Allow both API and mock prediction formats
export type AnyPrediction = ApiPrediction | MockPrediction;

// Type guard to check if prediction is API format
function isApiPrediction(pred: AnyPrediction | undefined): pred is ApiPrediction {
  return pred !== undefined && 'market_spread' in pred;
}

export interface ChatContext {
  game: GameDetail;
  prediction?: AnyPrediction;
  userBet?: UserBet;
}

export function buildGameContext(context: ChatContext | GameDetail): string {
  // Support both old GameDetail format and new ChatContext format
  const game = 'game' in context ? context.game : context;
  const apiPrediction = 'prediction' in context ? context.prediction : undefined;
  const userBet = 'userBet' in context ? context.userBet : undefined;
  
  const { home_team, away_team, home_score, away_score, status, quarter, time_remaining, possession } = game;
  const mockPrediction = (game as GameDetail).prediction;
  const market = (game as GameDetail).market;
  const homeStats = game.home_stats;
  const awayStats = game.away_stats;
  const homeQB = game.home_qb;
  const awayQB = game.away_qb;

  // Use API prediction if available, otherwise fall back to mock
  const prediction = apiPrediction || mockPrediction;

  // Format score and game state
  const scoreLine = `${away_team} ${away_score} @ ${home_team} ${home_score}`;
  const gameState = status === "Live" 
    ? `Q${quarter} — ${time_remaining || "N/A"} | ${possession === "home" ? home_team : away_team} ball`
    : status === "Scheduled"
    ? "Pre-game"
    : "Final";

  // Format team stats
  const homeStatsText = `${home_team}: ${homeStats.total_yards} total yards, ${homeStats.epa_per_play.toFixed(3)} EPA/play, ${homeStats.turnovers} turnovers`;
  const awayStatsText = `${away_team}: ${awayStats.total_yards} total yards, ${awayStats.epa_per_play.toFixed(3)} EPA/play, ${awayStats.turnovers} turnovers`;

  // Format prediction if available
  let predictionText = "";
  if (prediction) {
    // Handle both API Prediction and mock Prediction formats
    const winProbHome = 'win_prob_home' in prediction ? prediction.win_prob_home : (prediction as any).win_prob_home || 0.5;
    const predictedSpread = 'predicted_spread' in prediction ? prediction.predicted_spread : (prediction as any).predicted_spread || 0;
    const confidence = 'confidence' in prediction ? prediction.confidence : (prediction as any).confidence_score || 50;
    
    const winProbPercent = (winProbHome * 100).toFixed(1);
    const favoredTeam = predictedSpread < 0 ? home_team : away_team;
    const spreadAbs = Math.abs(predictedSpread);
    
    predictionText = `
Win Probability: ${winProbPercent}% ${home_team}
Model Spread: ${home_team} ${predictedSpread > 0 ? '+' : ''}${predictedSpread.toFixed(1)}
Confidence: ${confidence.toFixed(1)}%`;
    
    // Add market spread and edge if available
    if (isApiPrediction(apiPrediction)) {
      if (apiPrediction.market_spread !== null) {
        predictionText += `
Market Spread: ${home_team} ${apiPrediction.market_spread > 0 ? '+' : ''}${apiPrediction.market_spread.toFixed(1)}`;
      }
      if (apiPrediction.edge_vs_market !== null) {
        predictionText += `
Edge: ${apiPrediction.edge_vs_market > 0 ? '+' : ''}${apiPrediction.edge_vs_market.toFixed(1)} points`;
      }

      // Add top factors
      if (apiPrediction.top_factors && apiPrediction.top_factors.length > 0) {
        predictionText += `

## Top Factors Driving This Prediction
${apiPrediction.top_factors.slice(0, 5).map((f, i) =>
  `${i + 1}. **${f.description}**: ${f.value !== null ? f.value.toFixed(2) : 'N/A'} (${(f.importance * 100).toFixed(1)}% weight)`
).join('\n')}`;
      }
    } else if (market && mockPrediction) {
      const marketSpread = market.spread_home;
      const edge = (mockPrediction as any).edge_spread;
      predictionText += `
Market Spread: ${marketSpread.toFixed(1)}
Edge: ${edge > 0 ? "+" : ""}${edge.toFixed(1)} points`;
    }
  }

  // Format market if available
  let marketText = "";
  if (market) {
    marketText = `
Market Spread: ${market.spread_home.toFixed(1)} (${market.spread_home < 0 ? home_team : away_team} favored)
Market Total: ${market.total.toFixed(1)}`;
  }

  // Format QB stats
  let qbText = "";
  if (homeQB && awayQB) {
    const homeQBLine = `${home_team} QB: ${homeQB.name} — ${homeQB.completions}/${homeQB.attempts}, ${homeQB.yards} yds, ${homeQB.tds} TD, ${homeQB.ints} INT`;
    const awayQBLine = `${away_team} QB: ${awayQB.name} — ${awayQB.completions}/${awayQB.attempts}, ${awayQB.yards} yds, ${awayQB.tds} TD, ${awayQB.ints} INT`;
    
    // Add EPA if available
    const homeEPA = homeQB.epa !== undefined ? `, EPA: ${homeQB.epa.toFixed(2)}` : "";
    const awayEPA = awayQB.epa !== undefined ? `, EPA: ${awayQB.epa.toFixed(2)}` : "";
    
    qbText = `${awayQBLine}${awayEPA}\n${homeQBLine}${homeEPA}`;
  }

  // Format detailed stats
  let detailedStats = "";
  if (homeStats.third_down_conv && awayStats.third_down_conv) {
    detailedStats = `
Third Down Conversion:
  ${away_team}: ${awayStats.third_down_conv}
  ${home_team}: ${homeStats.third_down_conv}`;
    
    if (homeStats.red_zone_conv && awayStats.red_zone_conv) {
      detailedStats += `
Red Zone Conversion:
  ${away_team}: ${awayStats.red_zone_conv}
  ${home_team}: ${homeStats.red_zone_conv}`;
    }
  }

  // Format user bet if available
  let betText = "";
  if (userBet) {
    const currentMargin = home_score - away_score;
    const spreadNeeded = userBet.team === home_team ? -userBet.spread : userBet.spread;
    const covering = userBet.team === home_team 
      ? currentMargin > spreadNeeded
      : currentMargin < spreadNeeded;
    const coverMargin = userBet.team === home_team
      ? currentMargin - spreadNeeded
      : spreadNeeded - currentMargin;
    
    betText = `

## User's Bet
- **Position**: ${userBet.team} ${userBet.spread > 0 ? '+' : ''}${userBet.spread}${userBet.amount ? ` ($${userBet.amount})` : ''}
- **Current Status**: ${covering ? '✅ Covering' : '❌ Not covering'} by ${Math.abs(coverMargin).toFixed(1)} points
- **Needs**: ${userBet.team} to ${userBet.spread > 0 ? `lose by less than ${Math.abs(userBet.spread)} or win` : `win by more than ${Math.abs(userBet.spread)}`}`;
  }

  // Get win probability for game state analysis
  const winProbHome = prediction
    ? ('win_prob_home' in prediction ? prediction.win_prob_home : (prediction as any).win_prob_home || 0.5)
    : 0.5;

  // Analyze game state for contextual awareness
  const gameStateAnalysis = status === 'Live' || status === 'Final'
    ? analyzeGameState(home_team, away_team, home_score, away_score, quarter, time_remaining, winProbHome, status)
    : null;

  // Build game situation context for live games
  let situationContext = '';
  if (gameStateAnalysis && status === 'Live') {
    situationContext = `
## ⚠️ CRITICAL GAME SITUATION CONTEXT (READ THIS FIRST)
**Game Phase:** ${gameStateAnalysis.gamePhase.toUpperCase()} (${Math.round(gameStateAnalysis.minutesRemaining)} min remaining)
**Score Context:** ${gameStateAnalysis.situationSummary}
**Win Probability Meaning:** ${gameStateAnalysis.winProbInterpretation}
**Game Status:** ${gameStateAnalysis.gameCloseness.replace('_', ' ').toUpperCase()}

⚡ **YOUR RESPONSE MUST REFLECT THIS REALITY** - Do not overstate certainty. A ${Math.round(gameStateAnalysis.minutesRemaining)}-minute game with ${Math.abs(gameStateAnalysis.scoreDifferential)} point margin (${gameStateAnalysis.possessionsNeeded} score${gameStateAnalysis.possessionsNeeded > 1 ? 's' : ''}) is ${gameStateAnalysis.gameCloseness === 'coin_flip' || gameStateAnalysis.gameCloseness === 'competitive' ? 'VERY MUCH UNDECIDED' : gameStateAnalysis.gameCloseness === 'comfortable' ? 'leaning but not over' : gameStateAnalysis.gameCloseness === 'commanding' ? 'strongly favoring leader but possible to change' : 'likely decided'}.
`;
  }

  return `${situationContext}## Current Game
**${scoreLine}**
${gameState}

## Live Stats
| Team | Total Yds | EPA/Play | 3rd Down | Turnovers |
|------|-----------|----------|----------|-----------|
| ${away_team} | ${awayStats.total_yards || '—'} | ${awayStats.epa_per_play?.toFixed(2) || '—'} | ${awayStats.third_down_conv || '—'} | ${awayStats.turnovers || 0} |
| ${home_team} | ${homeStats.total_yards || '—'} | ${homeStats.epa_per_play?.toFixed(2) || '—'} | ${homeStats.third_down_conv || '—'} | ${homeStats.turnovers || 0} |

## Quarterbacks
- **${away_team}**: ${awayQB?.name || 'Unknown'} — ${awayQB?.completions || 0}/${awayQB?.attempts || 0}, ${awayQB?.yards || 0} yds, ${awayQB?.tds || 0} TD, ${awayQB?.ints || 0} INT
- **${home_team}**: ${homeQB?.name || 'Unknown'} — ${homeQB?.completions || 0}/${homeQB?.attempts || 0}, ${homeQB?.yards || 0} yds, ${homeQB?.tds || 0} TD, ${homeQB?.ints || 0} INT

## Model Prediction${predictionText}

## Market${marketText}${betText}`;
}

export function buildConversationContext(
  messages: Array<{ role: 'user' | 'assistant'; content: string }>,
  maxTurns: number = 4
): string {
  // Keep recent conversation for context (reduced from 6 to 4 turns)
  const recentMessages = messages.slice(-maxTurns * 2);
  
  if (recentMessages.length === 0) return '';
  
  return `

## Recent Conversation
${recentMessages.map(m => `**${m.role === 'user' ? 'User' : 'You'}**: ${m.content}`).join('\n\n')}

`;
}
