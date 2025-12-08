/**
 * Live Game Response Tester
 *
 * Simulates various game scenarios and tests Gemini's response alignment
 * with actual game state (score differential, time remaining, etc.)
 *
 * Usage: npx ts-node scripts/test-live-game-responses.ts
 */

import { buildGameContext, ChatContext } from '../src/lib/ai/context-builder';
import { GameDetail } from '../src/lib/mock_data';

// Test scenarios representing different game states
interface TestScenario {
  name: string;
  description: string;
  expectedTone: string; // What the AI response SHOULD convey
  game: Partial<GameDetail>;
  userMessage: string;
}

const baseGame: GameDetail = {
  game_id: 'test-jax-ind-2025',
  home_team: 'IND',
  away_team: 'JAX',
  home_score: 14,
  away_score: 21,
  status: 'Live',
  quarter: 3,
  time_remaining: '8:45',
  possession: 'home',
  home_stats: {
    total_yards: 245,
    plays: 42,
    epa_per_play: -0.05,
    third_down_conv: '4/10',
    red_zone_conv: '2/3',
    turnovers: 1,
  },
  away_stats: {
    total_yards: 312,
    plays: 48,
    epa_per_play: 0.12,
    third_down_conv: '6/11',
    red_zone_conv: '3/3',
    turnovers: 0,
  },
  home_qb: {
    name: 'Anthony Richardson',
    completions: 14,
    attempts: 22,
    yards: 185,
    tds: 1,
    ints: 1,
    qbr: 62.4,
    epa: -0.08,
  },
  away_qb: {
    name: 'Trevor Lawrence',
    completions: 18,
    attempts: 26,
    yards: 248,
    tds: 2,
    ints: 0,
    qbr: 84.2,
    epa: 0.18,
  },
  prediction: {
    win_prob_home: 0.35,
    predicted_spread: 3.5,
    confidence_score: 68,
    edge_spread: 1.2,
  },
  market: {
    spread_home: 2.5,
    spread_away: -2.5,
    total: 47.5,
    home_ml: 125,
    away_ml: -145,
  },
  win_probability: [],
  scoring_plays: [],
};

const testScenarios: TestScenario[] = [
  // Scenario 1: Close game, lots of time (current user issue)
  {
    name: '7-point game, Q3 (ONE SCORE, HALF GAME LEFT)',
    description: 'JAX leads by 7 with over a quarter remaining',
    expectedTone: 'Should emphasize this is a one-score game, very competitive, NOT a done deal',
    game: {
      home_score: 14,
      away_score: 21,
      quarter: 3,
      time_remaining: '8:45',
      prediction: { win_prob_home: 0.35, predicted_spread: 3.5, confidence_score: 68, edge_spread: 1.2 },
    },
    userMessage: 'Who do you think wins this game?',
  },

  // Scenario 2: Tied game
  {
    name: 'Tied game, Q4 (COIN FLIP)',
    description: 'Tied with 10 minutes left',
    expectedTone: 'Should say this is a true toss-up, could go either way',
    game: {
      home_score: 21,
      away_score: 21,
      quarter: 4,
      time_remaining: '10:00',
      prediction: { win_prob_home: 0.52, predicted_spread: -0.5, confidence_score: 52, edge_spread: 0.2 },
    },
    userMessage: 'What are the chances for each team?',
  },

  // Scenario 3: Two-score game, late
  {
    name: '14-point game, Q4 5min (TWO SCORES, GETTING DIFFICULT)',
    description: 'JAX leads by 14 with 5 minutes left',
    expectedTone: 'Should acknowledge IND needs two scoring drives in limited time - difficult but not impossible',
    game: {
      home_score: 14,
      away_score: 28,
      quarter: 4,
      time_remaining: '5:00',
      prediction: { win_prob_home: 0.12, predicted_spread: 10.5, confidence_score: 85, edge_spread: 2.1 },
    },
    userMessage: 'Can the Colts come back?',
  },

  // Scenario 4: Blowout
  {
    name: '24-point game, Q4 (BLOWOUT)',
    description: 'JAX leads by 24 with 8 minutes left',
    expectedTone: 'Should acknowledge game is effectively decided, would need historic comeback',
    game: {
      home_score: 7,
      away_score: 31,
      quarter: 4,
      time_remaining: '8:00',
      prediction: { win_prob_home: 0.02, predicted_spread: 18.5, confidence_score: 95, edge_spread: 4.2 },
    },
    userMessage: 'Is this game over?',
  },

  // Scenario 5: Close game, crunch time
  {
    name: '3-point game, 2min left (CRUNCH TIME)',
    description: 'JAX leads by 3 with under 2 minutes, IND has ball',
    expectedTone: 'Should emphasize every play matters, IND in position to tie/win with one good drive',
    game: {
      home_score: 24,
      away_score: 27,
      quarter: 4,
      time_remaining: '1:45',
      possession: 'home',
      prediction: { win_prob_home: 0.38, predicted_spread: 1.5, confidence_score: 62, edge_spread: 0.8 },
    },
    userMessage: 'What needs to happen for the Colts to win?',
  },

  // Scenario 6: Early game lead
  {
    name: '14-point game, Q1 (EARLY, LOTS OF TIME)',
    description: 'JAX leads by 14 but only Q1',
    expectedTone: 'Should note the lead but emphasize THREE QUARTERS of football remaining - way too early to call',
    game: {
      home_score: 0,
      away_score: 14,
      quarter: 1,
      time_remaining: '5:30',
      prediction: { win_prob_home: 0.25, predicted_spread: 7.5, confidence_score: 58, edge_spread: 1.5 },
    },
    userMessage: 'Jaguars are dominating! Is this going to be a blowout?',
  },

  // Scenario 7: 65% win probability interpretation
  {
    name: '65% win prob (EDGE, NOT A LOCK)',
    description: 'Model has JAX at 65%',
    expectedTone: 'Should explain 65% means JAX is favored but 35% chance IND wins - NOT a sure thing',
    game: {
      home_score: 17,
      away_score: 21,
      quarter: 3,
      time_remaining: '12:00',
      prediction: { win_prob_home: 0.35, predicted_spread: 2.5, confidence_score: 65, edge_spread: 1.0 },
    },
    userMessage: 'The model favors Jacksonville right?',
  },
];

function createTestGame(overrides: Partial<GameDetail>): GameDetail {
  return {
    ...baseGame,
    ...overrides,
    prediction: {
      ...baseGame.prediction,
      ...(overrides.prediction || {}),
    },
  } as GameDetail;
}

async function runTest(scenario: TestScenario) {
  console.log('\n' + '='.repeat(80));
  console.log(`SCENARIO: ${scenario.name}`);
  console.log('='.repeat(80));
  console.log(`Description: ${scenario.description}`);
  console.log(`Expected Tone: ${scenario.expectedTone}`);
  console.log(`User Message: "${scenario.userMessage}"`);
  console.log('-'.repeat(80));

  const game = createTestGame(scenario.game);
  const context: ChatContext = { game, prediction: game.prediction };
  const gameContext = buildGameContext(context);

  console.log('\nGAME CONTEXT SENT TO AI:');
  console.log(gameContext);
  console.log('-'.repeat(80));

  // If API key is available, actually call the API
  if (process.env.GEMINI_API_KEY) {
    try {
      const response = await fetch('http://localhost:3000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: scenario.userMessage,
          gameContext: context,
          conversationHistory: [],
        }),
      });

      if (response.ok && response.body) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullResponse = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value);
          const lines = chunk.split('\n').filter(l => l.startsWith('data: '));

          for (const line of lines) {
            const data = line.slice(6);
            if (data === '[DONE]') continue;
            try {
              const parsed = JSON.parse(data);
              if (parsed.content) fullResponse += parsed.content;
            } catch {}
          }
        }

        console.log('\nAI RESPONSE:');
        console.log(fullResponse);
        console.log('-'.repeat(80));

        // Basic validation
        const issues: string[] = [];

        if (scenario.name.includes('ONE SCORE') && fullResponse.toLowerCase().includes('control')) {
          issues.push('❌ Used "control" language for a one-score game');
        }
        if (scenario.name.includes('COIN FLIP') && fullResponse.toLowerCase().includes('favor')) {
          issues.push('❌ Used "favor" language for a coin flip');
        }
        if (scenario.name.includes('65%') && fullResponse.toLowerCase().includes('strongly')) {
          issues.push('❌ Used "strongly" for 65% probability');
        }

        if (issues.length > 0) {
          console.log('\n⚠️ POTENTIAL ISSUES:');
          issues.forEach(i => console.log(i));
        } else {
          console.log('\n✅ Response appears calibrated');
        }
      }
    } catch (error) {
      console.log('\n⚠️ Could not call API (server may not be running)');
      console.log('Context has been generated - review above to verify game state is properly communicated.');
    }
  } else {
    console.log('\n⚠️ No GEMINI_API_KEY - showing context only');
    console.log('Set GEMINI_API_KEY and run local server to test actual responses.');
  }
}

async function main() {
  console.log('🏈 LIVE GAME RESPONSE TESTER');
  console.log('Testing AI response calibration for various game scenarios\n');

  for (const scenario of testScenarios) {
    await runTest(scenario);
  }

  console.log('\n' + '='.repeat(80));
  console.log('TEST COMPLETE');
  console.log('='.repeat(80));
}

main().catch(console.error);
