import { NextRequest } from "next/server";
import { buildSystemPrompt } from "@/lib/ai/system-prompt";
import { MODEL_KNOWLEDGE } from "@/lib/ai/model-knowledge";
import { buildGameContext, ChatContext } from "@/lib/ai/context-builder";
import { streamLLMResponse, ChatMessage } from "@/lib/ai/providers";
import { GameDetail } from "@/lib/mock_data";
import { detectRefreshIntent, getRefreshPhrase } from "@/lib/ai/refresh-detector";
import { detectAdjustmentIntent, magnitudeToValue, describeAdjustment } from "@/lib/ai/adjustment-detector";
import { checkMessageSecurity, trackExtractionAttempt, getExtractionWarning, sanitizeUserInput } from "@/lib/ai/security";
import { refreshGameData, formatTimeSince } from "@/lib/services/game-refresh";
import { GameContext as AIGameContext, UserAdjustments, UserAdjustment } from "@/lib/ai/types";

// Use Node.js runtime (edge might have issues with some APIs)
// We'll handle streaming explicitly
export const runtime = "nodejs";

interface ChatRequest {
  message: string;
  conversationHistory?: ChatMessage[];
  gameContext?: ChatContext | GameDetail;
  userAdjustments?: UserAdjustments;
  sessionId?: string;
}

/**
 * Convert GameDetail or ChatContext to AIGameContext for system prompt
 */
function toAIGameContext(context: ChatContext | GameDetail | undefined): AIGameContext | null {
  if (!context) return null;

  // Handle ChatContext format
  if ('game' in context && context.game) {
    const game = context.game;
    const prediction = context.prediction;

    return {
      gameId: game.game_id,
      homeTeam: game.home_team,
      awayTeam: game.away_team,
      homeScore: game.home_score,
      awayScore: game.away_score,
      status: game.status === 'Live' ? 'live' : game.status === 'Final' ? 'final' : 'pregame',
      quarter: game.quarter,
      timeRemaining: game.time_remaining,
      possession: game.possession,
      prediction: {
        winProbHome: prediction?.win_prob_home ?? 0.5,
        predictedSpread: prediction?.predicted_spread ?? 0,
        marketSpread: ('market_spread' in (prediction || {})) ? (prediction as any).market_spread : null,
        edge: ('edge_vs_market' in (prediction || {})) ? (prediction as any).edge_vs_market : (prediction as any)?.edge_spread ?? null,
        confidence: ('confidence' in (prediction || {})) ? (prediction as any).confidence : (prediction as any)?.confidence_score ?? 50,
      },
      stats: game.home_stats && game.away_stats ? {
        home: {
          totalYards: game.home_stats.total_yards,
          totalPlays: game.home_stats.plays,
          epaPerPlay: game.home_stats.epa_per_play,
          thirdDownPct: parseThirdDown(game.home_stats.third_down_conv),
          redZonePct: parseRedZone(game.home_stats.red_zone_conv),
          turnovers: game.home_stats.turnovers,
        },
        away: {
          totalYards: game.away_stats.total_yards,
          totalPlays: game.away_stats.plays,
          epaPerPlay: game.away_stats.epa_per_play,
          thirdDownPct: parseThirdDown(game.away_stats.third_down_conv),
          redZonePct: parseRedZone(game.away_stats.red_zone_conv),
          turnovers: game.away_stats.turnovers,
        },
      } : undefined,
      qbStats: game.home_qb && game.away_qb ? {
        home: {
          name: game.home_qb.name,
          completions: game.home_qb.completions,
          attempts: game.home_qb.attempts,
          yards: game.home_qb.yards,
          touchdowns: game.home_qb.tds,
          interceptions: game.home_qb.ints,
          epaPerPlay: game.home_qb.epa,
        },
        away: {
          name: game.away_qb.name,
          completions: game.away_qb.completions,
          attempts: game.away_qb.attempts,
          yards: game.away_qb.yards,
          touchdowns: game.away_qb.tds,
          interceptions: game.away_qb.ints,
          epaPerPlay: game.away_qb.epa,
        },
      } : undefined,
    };
  }

  // Handle direct GameDetail format
  const game = context as GameDetail;
  return {
    gameId: game.game_id,
    homeTeam: game.home_team,
    awayTeam: game.away_team,
    homeScore: game.home_score,
    awayScore: game.away_score,
    status: game.status === 'Live' ? 'live' : game.status === 'Final' ? 'final' : 'pregame',
    quarter: game.quarter,
    timeRemaining: game.time_remaining,
    possession: game.possession,
    prediction: {
      winProbHome: game.prediction?.win_prob_home ?? 0.5,
      predictedSpread: game.prediction?.predicted_spread ?? 0,
      marketSpread: game.market?.spread_home ?? null,
      edge: game.prediction?.edge_spread ?? null,
      confidence: game.prediction?.confidence_score ?? 50,
    },
    stats: game.home_stats && game.away_stats ? {
      home: {
        totalYards: game.home_stats.total_yards,
        totalPlays: game.home_stats.plays,
        epaPerPlay: game.home_stats.epa_per_play,
        thirdDownPct: parseThirdDown(game.home_stats.third_down_conv),
        redZonePct: parseRedZone(game.home_stats.red_zone_conv),
        turnovers: game.home_stats.turnovers,
      },
      away: {
        totalYards: game.away_stats.total_yards,
        totalPlays: game.away_stats.plays,
        epaPerPlay: game.away_stats.epa_per_play,
        thirdDownPct: parseThirdDown(game.away_stats.third_down_conv),
        redZonePct: parseRedZone(game.away_stats.red_zone_conv),
        turnovers: game.away_stats.turnovers,
      },
    } : undefined,
    qbStats: game.home_qb && game.away_qb ? {
      home: {
        name: game.home_qb.name,
        completions: game.home_qb.completions,
        attempts: game.home_qb.attempts,
        yards: game.home_qb.yards,
        touchdowns: game.home_qb.tds,
        interceptions: game.home_qb.ints,
        epaPerPlay: game.home_qb.epa,
      },
      away: {
        name: game.away_qb.name,
        completions: game.away_qb.completions,
        attempts: game.away_qb.attempts,
        yards: game.away_qb.yards,
        touchdowns: game.away_qb.tds,
        interceptions: game.away_qb.ints,
        epaPerPlay: game.away_qb.epa,
      },
    } : undefined,
  };
}

function parseThirdDown(conv: string | undefined): number {
  if (!conv) return 0;
  const [made, att] = conv.split('/').map(Number);
  return att > 0 ? Math.round((made / att) * 100) : 0;
}

function parseRedZone(conv: string | undefined): number {
  if (!conv) return 0;
  const [made, att] = conv.split('/').map(Number);
  return att > 0 ? Math.round((made / att) * 100) : 0;
}

export async function POST(request: NextRequest) {
  try {
    const body: ChatRequest = await request.json();
    const {
      message,
      conversationHistory = [],
      gameContext,
      userAdjustments,
      sessionId = 'default'
    } = body;

    if (!message || typeof message !== "string") {
      return new Response(
        JSON.stringify({ error: "Message is required" }),
        { status: 400, headers: { "Content-Type": "application/json" } }
      );
    }

    // Sanitize user input
    const sanitizedMessage = sanitizeUserInput(message);

    // Security check
    const securityCheck = checkMessageSecurity(sanitizedMessage);
    if (securityCheck.isExtractionAttempt || securityCheck.isSuspicious) {
      const attemptCount = trackExtractionAttempt(sessionId);
      const warning = getExtractionWarning(attemptCount);

      if (warning && securityCheck.severity === 'high') {
        // Return a polite deflection for severe attempts
        return new Response(
          JSON.stringify({ content: warning, done: true }),
          { headers: { "Content-Type": "application/json" } }
        );
      }
    }

    // Detect intents
    const refreshIntent = detectRefreshIntent(sanitizedMessage);
    const adjustmentIntent = detectAdjustmentIntent(sanitizedMessage);

    // Handle refresh if detected and we have a game context
    let refreshResult = null;
    const gameId = gameContext
      ? ('game' in gameContext && gameContext.game?.game_id)
        ? gameContext.game.game_id
        : (gameContext as GameDetail).game_id
      : null;

    if (refreshIntent.detected && gameId) {
      refreshResult = await refreshGameData(gameId);
    }

    // Convert to AI game context format
    const aiGameContext = toAIGameContext(gameContext);

    // Build system prompt with all context
    const systemPrompt = buildSystemPrompt(
      MODEL_KNOWLEDGE,
      aiGameContext,
      userAdjustments || null
    );

    // Build game context string for the user message
    let gameContextText = "";
    if (gameContext) {
      gameContextText = buildGameContext(gameContext);
    }

    // Prepare messages - limit conversation history
    const recentHistory = conversationHistory.slice(-8).map(m => ({
      role: m.role as "user" | "assistant",
      content: m.content,
    }));

    // Build enhanced user message with context
    let enhancedMessage = "";

    // Add refresh context if applicable
    if (refreshResult) {
      if (refreshResult.refreshed) {
        if (refreshResult.changesDetected) {
          enhancedMessage += `[CONTEXT: Data refresh completed. Changes: ${refreshResult.changes?.join(', ')}. Acknowledge naturally.]\n\n`;
        } else {
          enhancedMessage += `[CONTEXT: Data refresh completed. No changes since ${formatTimeSince(refreshResult.lastUpdated)}. Confirm data is current.]\n\n`;
        }
      } else if (refreshResult.error) {
        enhancedMessage += `[CONTEXT: Refresh note: ${refreshResult.error}. Handle gracefully.]\n\n`;
      }
    }

    // Add adjustment context if applicable
    if (adjustmentIntent.detected) {
      if (adjustmentIntent.category === 'reset') {
        enhancedMessage += `[CONTEXT: User requested reset to default weights. Confirm the reset and show original prediction values.]\n\n`;
      } else if (adjustmentIntent.direction) {
        const magnitude = magnitudeToValue(adjustmentIntent.magnitude);
        const description = describeAdjustment(adjustmentIntent);
        enhancedMessage += `[CONTEXT: User requested adjustment - ${description}. Magnitude: ${(magnitude * 100).toFixed(0)}%, Scope: ${adjustmentIntent.scope}. Acknowledge, explain impact, show before/after.]\n\n`;
      }
    }

    // Add game context and user message
    if (gameContextText) {
      enhancedMessage += `${gameContextText}\n\n`;
    }
    enhancedMessage += sanitizedMessage;

    const messages: ChatMessage[] = [
      ...recentHistory,
      { role: "user", content: enhancedMessage },
    ];

    // Validate messages
    if (messages.length === 0 || !messages.some(m => m.role === "user")) {
      return new Response(
        JSON.stringify({ error: "Invalid message format" }),
        { status: 400, headers: { "Content-Type": "application/json" } }
      );
    }

    // Get API key
    const apiKey = process.env.GEMINI_API_KEY || process.env.GOOGLE_AI_API_KEY || process.env.OPENAI_API_KEY;

    if (!apiKey) {
      return new Response(
        JSON.stringify({ error: "API key not configured. Please set GEMINI_API_KEY in .env.local" }),
        { status: 500, headers: { "Content-Type": "application/json" } }
      );
    }

    // Create streaming response - ensure immediate flushing
    const encoder = new TextEncoder();
    let hasContent = false;
    let chunkCount = 0;

    const stream = new ReadableStream({
      async start(controller) {
        try {
          let totalContentLength = 0;
          let lastChunkTime = Date.now();
          
          for await (const chunk of streamLLMResponse(messages, systemPrompt, apiKey)) {
            if (chunk.content) {
              hasContent = true;
              chunkCount++;
              totalContentLength += chunk.content.length;
              lastChunkTime = Date.now();
              // Enqueue immediately - chunks contain 2 characters each for faster streaming
              // No delays here to ensure first token arrives immediately
              const data = `data: ${JSON.stringify({ content: chunk.content })}\n\n`;
              controller.enqueue(encoder.encode(data));
            }
            if (chunk.done) {
              console.log(`[API Route] Stream done. Total chunks: ${chunkCount}, Total content length: ${totalContentLength}`);
              console.log(`[API Route] Content preview: "${totalContentLength > 0 ? '...' : ''}"`);
              if (!hasContent) {
                console.warn("[API Route] Stream completed but no content was received");
              }
              // Ensure we send [DONE] signal
              controller.enqueue(encoder.encode(`data: [DONE]\n\n`));
              controller.close();
              return;
            }
          }
          
          // If we exit the loop without a done signal, check if we got content
          if (!hasContent) {
            console.warn("[API Route] Stream ended without content or done signal");
          } else {
            console.log(`[API Route] Stream loop ended. Total chunks: ${chunkCount}, Total content length: ${totalContentLength}`);
          }
          
          // Ensure we send [DONE] signal even if done wasn't explicitly set
          controller.enqueue(encoder.encode(`data: [DONE]\n\n`));
          // Minimal delay only for final flush - doesn't affect first token
          await new Promise(resolve => setTimeout(resolve, 50));
          controller.close();
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : "Unknown error";
          console.error("[API Route] Chat API error:", errorMessage, error);
          controller.enqueue(
            encoder.encode(`data: ${JSON.stringify({ error: errorMessage })}\n\n`)
          );
          // Send [DONE] even on error to properly close the stream
          controller.enqueue(encoder.encode(`data: [DONE]\n\n`));
          await new Promise(resolve => setTimeout(resolve, 50));
          controller.close();
        }
      },
    });

    return new Response(stream, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no", // Disable nginx buffering
      },
    });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : "Unknown error";
    return new Response(
      JSON.stringify({ error: errorMessage }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    );
  }
}
