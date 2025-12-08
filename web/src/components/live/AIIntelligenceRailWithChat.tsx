"use client";

import { useState, useEffect } from "react";
import { GameDetail } from "@/lib/mock_data";
import { cn } from "@/lib/utils";
import { motion, AnimatePresence } from "framer-motion";
import {
    Brain,
    CheckCircle,
    AlertCircle,
    TrendingUp,
    Activity,
    Sparkles,
    MessageSquare,
    ChevronDown,
    ChevronUp,
} from "lucide-react";
import { AIChat } from "@/components/chat/AIChat";
import { getPrediction, Prediction } from "@/lib/api/predictions";
import { ChatContext } from "@/lib/ai/context-builder";

interface AIIntelligenceRailWithChatProps {
    game: GameDetail;
}

export function AIIntelligenceRailWithChat({ game }: AIIntelligenceRailWithChatProps) {
    const [showChat, setShowChat] = useState(false);
    const [apiPrediction, setApiPrediction] = useState<Prediction | null>(null);
    const [predictionLoading, setPredictionLoading] = useState(false);
    const [predictionError, setPredictionError] = useState<string | null>(null);
    
    const mockPrediction = game.prediction;
    const market = game.market;
    const isLive = game.status === "Live";
    
    // Use API prediction if available, otherwise fall back to mock
    const prediction = apiPrediction || mockPrediction;
    
    // Fetch real prediction when game changes
    useEffect(() => {
        const fetchPrediction = async () => {
            setPredictionLoading(true);
            setPredictionError(null);
            try {
                const pred = await getPrediction(game.game_id);
                setApiPrediction(pred);
            } catch (error) {
                // Only log unexpected errors (404s return null, not throw)
                console.warn("Failed to fetch prediction:", error);
                setPredictionError(error instanceof Error ? error.message : "Failed to fetch prediction");
                setApiPrediction(null);
            } finally {
                setPredictionLoading(false);
            }
        };
        
        fetchPrediction();
    }, [game.game_id]);

    // Calculate metrics - handle both API and mock prediction formats
    const modelEdge = apiPrediction 
        ? (apiPrediction.edge_vs_market || 0)
        : (prediction as any)?.edge_spread || 0;

    // Determine recommendation
    const predictedSpread = apiPrediction 
        ? apiPrediction.predicted_spread 
        : (prediction as any)?.predicted_spread || 0;
    const modelFavorsHome = predictedSpread < 0;
    const favoredTeam = modelFavorsHome ? game.home_team : game.away_team;
    const absEdge = Math.abs(modelEdge);

    // Generate key factors and reasoning
    const keyFactors = generateKeyFactors(game, prediction, market);
    const modelReasoning = generateModelReasoning(game, prediction, market);

    return (
        <motion.div
            className="card-secondary p-4 h-full flex flex-col"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
        >
            {!showChat ? (
                // Summary View
                <>
                    {/* Header */}
                    <div className="flex items-center justify-between mb-3 flex-shrink-0">
                        <div className="flex items-center gap-2">
                            <Brain size={14} className="text-emerald-400" />
                            <span className="text-xs font-medium text-white/60">AI Insights</span>
                        </div>
                        <div className="flex items-center gap-2">
                            {isLive && (
                                <div className="flex items-center gap-1.5">
                                    <Activity size={10} className="text-emerald-400" />
                                    <span className="text-[9px] text-emerald-400 uppercase tracking-wider font-medium">
                                        Live
                                    </span>
                                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                                </div>
                            )}
                            <button
                                onClick={() => setShowChat(true)}
                                className="p-1.5 rounded-lg hover:bg-white/[0.05] transition-colors flex items-center gap-1.5"
                            >
                                <MessageSquare size={12} className="text-white/50" />
                                <span className="text-[10px] text-white/50">Chat</span>
                            </button>
                        </div>
                    </div>

                    {/* Primary Recommendation */}
                    {absEdge >= 1 && (
                        <div className={cn(
                            "p-3 rounded-lg mb-3 border flex-shrink-0",
                            modelEdge > 0
                                ? "bg-emerald-500/5 border-emerald-500/10"
                                : "bg-amber-500/5 border-amber-500/10"
                        )}>
                            <div className={cn(
                                "text-sm font-medium",
                                modelEdge > 0 ? "text-emerald-400" : "text-amber-400"
                            )}>
                                Take {favoredTeam} {market?.spread_home ? (
                                    market.spread_home > 0 ? `+${market.spread_home}` : market.spread_home
                                ) : ""}
                            </div>
                            <div className="text-xs text-white/50 mt-1">
                                Model shows +{absEdge.toFixed(1)} edge. {absEdge >= 2 ? "Strong" : absEdge >= 1.5 ? "Moderate" : "Small"} confidence.
                            </div>
                        </div>
                    )}

                    {/* Key Factors */}
                    <div className="space-y-2 flex-shrink-0 mb-4">
                        <span className="text-[9px] uppercase tracking-wider text-white/30 font-medium">
                            Key Factors
                        </span>

                        {keyFactors.slice(0, 3).map((factor, index) => (
                            <div key={index} className="flex items-start gap-2">
                                {factor.type === "positive" ? (
                                    <CheckCircle className="w-3 h-3 text-emerald-400 mt-0.5 flex-shrink-0" />
                                ) : factor.type === "negative" ? (
                                    <AlertCircle className="w-3 h-3 text-amber-400 mt-0.5 flex-shrink-0" />
                                ) : (
                                    <TrendingUp className="w-3 h-3 text-blue-400 mt-0.5 flex-shrink-0" />
                                )}
                                <span className="text-xs text-white/70 leading-tight">{factor.message}</span>
                            </div>
                        ))}
                    </div>

                    {/* Model Reasoning Summary - Fills remaining space */}
                    <div className="flex-1 min-h-0 flex flex-col">
                        <div className="flex items-center gap-2 mb-2 flex-shrink-0">
                            <Sparkles size={12} className="text-purple-400" />
                            <span className="text-[9px] uppercase tracking-wider text-white/30 font-medium">
                                Model Analysis
                            </span>
                        </div>
                        <div className="flex-1 overflow-y-auto scrollbar-chat">
                            <div className="text-xs text-white/60 leading-relaxed space-y-2">
                                {modelReasoning.map((paragraph, index) => (
                                    <p key={index}>{paragraph}</p>
                                ))}
                            </div>
                        </div>
                    </div>
                </>
            ) : (
                // Chat View
                <div className="flex-1 min-h-0 flex flex-col">
                    <div className="flex items-center justify-between mb-3 flex-shrink-0">
                        <div className="flex items-center gap-2">
                            <Brain size={14} className="text-emerald-400" />
                            <span className="text-xs font-medium text-white/60">AI Chat</span>
                        </div>
                        <button
                            onClick={() => setShowChat(false)}
                            className="p-1.5 rounded-lg hover:bg-white/[0.05] transition-colors flex items-center gap-1.5"
                        >
                            <ChevronUp size={12} className="text-white/50" />
                            <span className="text-[10px] text-white/50">Summary</span>
                        </button>
                    </div>
                    <div className="flex-1 min-h-0">
                        <AIChat 
                            game={game}
                            gameContext={{
                                game,
                                prediction: apiPrediction || undefined,
                            }}
                        />
                    </div>
                </div>
            )}
        </motion.div>
    );
}

// Union type for both API and mock predictions
type AnyPrediction = Prediction | GameDetail["prediction"];

// Helper functions (same as original AIIntelligenceRail)
function generateKeyFactors(
    game: GameDetail,
    prediction: AnyPrediction,
    market: GameDetail["market"]
): Array<{ message: string; type: "positive" | "negative" | "neutral" }> {
    const factors: Array<{ message: string; type: "positive" | "negative" | "neutral" }> = [];

    if (game.home_stats.epa_per_play > 0.1) {
        factors.push({
            message: `${game.home_team} offensive EPA +${game.home_stats.epa_per_play.toFixed(2)}/play`,
            type: "positive"
        });
    } else if (game.away_stats.epa_per_play > 0.1) {
        factors.push({
            message: `${game.away_team} offensive EPA +${game.away_stats.epa_per_play.toFixed(2)}/play`,
            type: "positive"
        });
    }

    if (game.home_qb && game.home_qb.qbr < 50) {
        factors.push({
            message: `${game.home_qb.name} struggling (${game.home_qb.qbr.toFixed(1)} QBR)`,
            type: "negative"
        });
    } else if (game.away_qb && game.away_qb.qbr < 50) {
        factors.push({
            message: `${game.away_qb.name} struggling (${game.away_qb.qbr.toFixed(1)} QBR)`,
            type: "negative"
        });
    }

    const turnoverDiff = game.away_stats.turnovers - game.home_stats.turnovers;
    if (Math.abs(turnoverDiff) >= 2) {
        factors.push({
            message: `Turnover diff: ${turnoverDiff > 0 ? game.home_team : game.away_team} +${Math.abs(turnoverDiff)}`,
            type: turnoverDiff > 0 ? "positive" : "negative"
        });
    }

    if (factors.length === 0) {
        factors.push({
            message: "Analysis in progress",
            type: "neutral"
        });
    }

    return factors.slice(0, 4);
}

function generateModelReasoning(
    game: GameDetail,
    prediction: AnyPrediction,
    market: GameDetail["market"]
): string[] {
    const reasoning: string[] = [];
    const modelFavorsHome = prediction ? prediction.predicted_spread < 0 : false;
    const favoredTeam = modelFavorsHome ? game.home_team : game.away_team;
    // Handle both API (edge_vs_market) and mock (edge_spread) prediction types
    const edge = prediction
        ? ('edge_vs_market' in prediction ? prediction.edge_vs_market : (prediction as any)?.edge_spread) || 0
        : 0;
    const absEdge = Math.abs(edge);

    if (absEdge >= 1) {
        reasoning.push(
            `The model identifies ${favoredTeam} as the stronger play in this matchup, projecting a ${absEdge.toFixed(1)}-point edge over the current market line.`
        );
    } else {
        reasoning.push(
            `This game projects close to market expectations with minimal edge detected. The model sees this as a fairly priced matchup.`
        );
    }

    const homeQBR = game.home_qb?.qbr || 0;
    const awayQBR = game.away_qb?.qbr || 0;
    if (Math.abs(homeQBR - awayQBR) > 15) {
        const betterQB = homeQBR > awayQBR ? game.home_qb : game.away_qb;
        const worseQB = homeQBR > awayQBR ? game.away_qb : game.home_qb;
        reasoning.push(
            `Quarterback play is a significant factor: ${betterQB?.name} (${betterQB?.qbr.toFixed(1)} QBR) has been substantially more efficient than ${worseQB?.name} (${worseQB?.qbr.toFixed(1)} QBR), impacting scoring projections.`
        );
    }

    const homeEPA = game.home_stats.epa_per_play;
    const awayEPA = game.away_stats.epa_per_play;
    if (Math.abs(homeEPA - awayEPA) > 0.1) {
        const betterTeam = homeEPA > awayEPA ? game.home_team : game.away_team;
        const betterEPA = Math.max(homeEPA, awayEPA);
        reasoning.push(
            `${betterTeam}'s offensive efficiency (${betterEPA > 0 ? '+' : ''}${betterEPA.toFixed(2)} EPA/play) suggests they are generating more valuable plays on a per-snap basis.`
        );
    }

    const toDiff = game.away_stats.turnovers - game.home_stats.turnovers;
    if (Math.abs(toDiff) >= 2) {
        const protectingTeam = toDiff > 0 ? game.home_team : game.away_team;
        reasoning.push(
            `Ball security favors ${protectingTeam} with a ${Math.abs(toDiff)}-turnover differential. This unsustainable variance may regress, but currently impacts win probability.`
        );
    }

    if (absEdge >= 2) {
        reasoning.push(
            `Given these factors, the model rates this as a high-confidence opportunity where the market may be undervaluing ${favoredTeam}'s advantages.`
        );
    } else if (absEdge >= 1) {
        reasoning.push(
            `While not a slam dunk, the model sees enough of an edge to warrant consideration if aligned with your betting strategy.`
        );
    }

    return reasoning;
}

