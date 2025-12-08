"use client";

import { GameDetail } from "@/lib/mock_data";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
import {
    Brain,
    CheckCircle,
    AlertCircle,
    TrendingUp,
    Activity,
    Sparkles,
} from "lucide-react";

interface AIIntelligenceRailProps {
    game: GameDetail;
}

export function AIIntelligenceRail({ game }: AIIntelligenceRailProps) {
    const prediction = game.prediction;
    const market = game.market;
    const isLive = game.status === "Live";

    // Calculate metrics
    const modelEdge = prediction?.edge_spread || 0;

    // Determine recommendation
    const modelFavorsHome = prediction ? prediction.predicted_spread < 0 : false;
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
            {/* Header */}
            <div className="flex items-center justify-between mb-3 flex-shrink-0">
                <div className="flex items-center gap-2">
                    <Brain size={14} className="text-emerald-400" />
                    <span className="text-xs font-medium text-white/60">AI Insights</span>
                </div>
                {isLive && (
                    <div className="flex items-center gap-1.5">
                        <Activity size={10} className="text-emerald-400" />
                        <span className="text-[9px] text-emerald-400 uppercase tracking-wider font-medium">
                            Live
                        </span>
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                    </div>
                )}
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
                <div className="flex-1 overflow-y-auto">
                    <div className="text-xs text-white/60 leading-relaxed space-y-2">
                        {modelReasoning.map((paragraph, index) => (
                            <p key={index}>{paragraph}</p>
                        ))}
                    </div>
                </div>
            </div>
        </motion.div>
    );
}

// Helper function to generate key factors
function generateKeyFactors(
    game: GameDetail,
    prediction: GameDetail["prediction"],
    market: GameDetail["market"]
): Array<{ message: string; type: "positive" | "negative" | "neutral" }> {
    const factors: Array<{ message: string; type: "positive" | "negative" | "neutral" }> = [];

    // EPA-based insight
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

    // QB performance insight
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

    // Turnover differential
    const turnoverDiff = game.away_stats.turnovers - game.home_stats.turnovers;
    if (Math.abs(turnoverDiff) >= 2) {
        factors.push({
            message: `Turnover diff: ${turnoverDiff > 0 ? game.home_team : game.away_team} +${Math.abs(turnoverDiff)}`,
            type: turnoverDiff > 0 ? "positive" : "negative"
        });
    }

    // Default if no factors
    if (factors.length === 0) {
        factors.push({
            message: "Analysis in progress",
            type: "neutral"
        });
    }

    return factors.slice(0, 4);
}

// Helper function to generate model reasoning summary
function generateModelReasoning(
    game: GameDetail,
    prediction: GameDetail["prediction"],
    market: GameDetail["market"]
): string[] {
    const reasoning: string[] = [];
    const modelFavorsHome = prediction ? prediction.predicted_spread < 0 : false;
    const favoredTeam = modelFavorsHome ? game.home_team : game.away_team;
    const underdogTeam = modelFavorsHome ? game.away_team : game.home_team;
    const edge = prediction?.edge_spread || 0;
    const absEdge = Math.abs(edge);

    // Opening reasoning
    if (absEdge >= 1) {
        reasoning.push(
            `The model identifies ${favoredTeam} as the stronger play in this matchup, projecting a ${absEdge.toFixed(1)}-point edge over the current market line.`
        );
    } else {
        reasoning.push(
            `This game projects close to market expectations with minimal edge detected. The model sees this as a fairly priced matchup.`
        );
    }

    // QB analysis
    const homeQBR = game.home_qb?.qbr || 0;
    const awayQBR = game.away_qb?.qbr || 0;
    if (Math.abs(homeQBR - awayQBR) > 15) {
        const betterQB = homeQBR > awayQBR ? game.home_qb : game.away_qb;
        const worseQB = homeQBR > awayQBR ? game.away_qb : game.home_qb;
        reasoning.push(
            `Quarterback play is a significant factor: ${betterQB?.name} (${betterQB?.qbr.toFixed(1)} QBR) has been substantially more efficient than ${worseQB?.name} (${worseQB?.qbr.toFixed(1)} QBR), impacting scoring projections.`
        );
    }

    // EPA analysis
    const homeEPA = game.home_stats.epa_per_play;
    const awayEPA = game.away_stats.epa_per_play;
    if (Math.abs(homeEPA - awayEPA) > 0.1) {
        const betterTeam = homeEPA > awayEPA ? game.home_team : game.away_team;
        const betterEPA = Math.max(homeEPA, awayEPA);
        reasoning.push(
            `${betterTeam}'s offensive efficiency (${betterEPA > 0 ? '+' : ''}${betterEPA.toFixed(2)} EPA/play) suggests they are generating more valuable plays on a per-snap basis.`
        );
    }

    // Turnover analysis
    const toDiff = game.away_stats.turnovers - game.home_stats.turnovers;
    if (Math.abs(toDiff) >= 2) {
        const protectingTeam = toDiff > 0 ? game.home_team : game.away_team;
        reasoning.push(
            `Ball security favors ${protectingTeam} with a ${Math.abs(toDiff)}-turnover differential. This unsustainable variance may regress, but currently impacts win probability.`
        );
    }

    // Closing summary
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

export default AIIntelligenceRail;
