"use client";

import { GameDetail } from "@/lib/mock_data";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
import { 
    Brain, 
    TrendingUp, 
    TrendingDown, 
    AlertTriangle, 
    Zap, 
    Target,
    Activity,
    BarChart3
} from "lucide-react";
import { ConfidenceMeter, EdgeIndicator, AIInsight, GlassDivider } from "@/components/ui/glass";

interface AIIntelligenceRailProps {
    game: GameDetail;
}

export function AIIntelligenceRail({ game }: AIIntelligenceRailProps) {
    const prediction = game.prediction;
    const market = game.market;
    const isLive = game.status === "Live";
    
    // Calculate metrics
    const confidence = prediction?.confidence_score || 50;
    const modelEdge = prediction?.edge_spread || 0;
    const totalEdge = prediction?.edge_total || 0;
    const winProb = prediction?.win_prob_home || 0.5;
    
    // Determine volatility (mock calculation based on win prob distance from 50%)
    const volatility = Math.abs(winProb - 0.5) * 100;
    const volatilityLabel = volatility > 30 ? "High" : volatility > 15 ? "Medium" : "Low";
    
    // Generate insights based on game state
    const insights = generateInsights(game, prediction, market);
    
    // Alerts
    const alerts = generateAlerts(game, prediction, market);

    return (
        <motion.div
            className="glass-rail p-4 ai-rail"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
        >
            {/* Header */}
            <div className="flex items-center gap-2 mb-5">
                <div className="p-2 rounded-lg bg-[var(--neon-blue-muted)] border border-[var(--neon-blue)]/20 ai-scan-effect">
                    <Brain size={16} className="text-[var(--neon-blue)]" />
                </div>
                <div>
                    <h3 className="text-sm font-semibold text-[var(--foreground)]">
                        AI Co-Pilot
                    </h3>
                    <p className="text-[10px] text-[var(--muted-foreground)]">
                        Real-time model intelligence
                    </p>
                </div>
            </div>

            {/* Model Confidence Section */}
            <div className="space-y-4 mb-5">
                <div className="flex items-center gap-2">
                    <Target size={12} className="text-[var(--muted-foreground)]" />
                    <span className="text-[11px] uppercase tracking-wider text-[var(--muted-foreground)] font-semibold">
                        Model Confidence
                    </span>
                </div>
                
                <ConfidenceMeter 
                    value={confidence} 
                    label=""
                    showValue={true}
                />
                
                <div className="grid grid-cols-2 gap-2">
                    <div className="glass-inner p-2.5 text-center">
                        <div className="text-[9px] text-[var(--muted-foreground)] uppercase tracking-wide">
                            Volatility
                        </div>
                        <div className={cn(
                            "text-sm font-semibold mt-0.5",
                            volatilityLabel === "High" && "text-[var(--warning)]",
                            volatilityLabel === "Medium" && "text-[var(--neon-blue)]",
                            volatilityLabel === "Low" && "text-[var(--neon-green)]"
                        )}>
                            {volatilityLabel}
                        </div>
                    </div>
                    <div className="glass-inner p-2.5 text-center">
                        <div className="text-[9px] text-[var(--muted-foreground)] uppercase tracking-wide">
                            Certainty
                        </div>
                        <div className="text-sm font-semibold text-[var(--foreground)] mt-0.5">
                            {confidence >= 80 ? "Very High" : confidence >= 60 ? "High" : confidence >= 40 ? "Moderate" : "Low"}
                        </div>
                    </div>
                </div>
            </div>

            <GlassDivider glow className="my-5" />

            {/* Edge Analysis */}
            <div className="space-y-4 mb-5">
                <div className="flex items-center gap-2">
                    <BarChart3 size={12} className="text-[var(--muted-foreground)]" />
                    <span className="text-[11px] uppercase tracking-wider text-[var(--muted-foreground)] font-semibold">
                        Edge Analysis
                    </span>
                </div>
                
                <EdgeIndicator 
                    value={modelEdge} 
                    label="Spread Edge"
                />
                
                <EdgeIndicator 
                    value={totalEdge} 
                    label="Total Edge"
                />
                
                {/* Market vs Model Summary */}
                <div className="glass-inner p-3">
                    <div className="flex items-center justify-between mb-2">
                        <span className="text-[10px] text-[var(--muted-foreground)]">Market Line</span>
                        <span className="text-xs font-mono font-semibold text-[var(--foreground)]">
                            {market?.spread_home ? (market.spread_home > 0 ? "+" : "") + market.spread_home : "--"}
                        </span>
                    </div>
                    <div className="flex items-center justify-between">
                        <span className="text-[10px] text-[var(--muted-foreground)]">Model Line</span>
                        <span className="text-xs font-mono font-semibold text-[var(--neon-blue)]">
                            {prediction?.predicted_spread ? (prediction.predicted_spread > 0 ? "+" : "") + prediction.predicted_spread.toFixed(1) : "--"}
                        </span>
                    </div>
                </div>
            </div>

            <GlassDivider glow className="my-5" />

            {/* Alerts Section */}
            {alerts.length > 0 && (
                <div className="space-y-3 mb-5">
                    <div className="flex items-center gap-2">
                        <AlertTriangle size={12} className="text-[var(--warning)]" />
                        <span className="text-[11px] uppercase tracking-wider text-[var(--muted-foreground)] font-semibold">
                            Alerts
                        </span>
                    </div>
                    
                    {alerts.map((alert, index) => (
                        <motion.div
                            key={index}
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: 0.1 * index }}
                            className={cn(
                                "glass-inner p-3 border-l-2",
                                alert.severity === "high" && "border-l-[var(--warning)]",
                                alert.severity === "medium" && "border-l-[var(--neon-blue)]",
                                alert.severity === "low" && "border-l-[var(--muted-foreground)]"
                            )}
                        >
                            <p className="text-[11px] text-[var(--foreground)] leading-relaxed">
                                {alert.message}
                            </p>
                        </motion.div>
                    ))}
                </div>
            )}

            {alerts.length > 0 && <GlassDivider className="my-5" />}

            {/* Insights Section */}
            <div className="space-y-3">
                <div className="flex items-center gap-2">
                    <Zap size={12} className="text-[var(--neon-blue)]" />
                    <span className="text-[11px] uppercase tracking-wider text-[var(--muted-foreground)] font-semibold">
                        Key Insights
                    </span>
                </div>
                
                {insights.map((insight, index) => (
                    <AIInsight
                        key={index}
                        message={insight.message}
                        timestamp={insight.timestamp}
                        type={insight.type}
                    />
                ))}
            </div>

            {/* Live Indicator */}
            {isLive && (
                <>
                    <GlassDivider className="my-5" />
                    <div className="flex items-center justify-center gap-2 py-2">
                        <Activity size={12} className="text-[var(--neon-green)]" />
                        <span className="text-[10px] text-[var(--neon-green)] uppercase tracking-wider font-semibold">
                            Updating Live
                        </span>
                        <span className="live-indicator" />
                    </div>
                </>
            )}
        </motion.div>
    );
}

// Helper functions to generate dynamic content
function generateInsights(
    game: GameDetail, 
    prediction: GameDetail["prediction"], 
    market: GameDetail["market"]
): Array<{ message: string; timestamp?: string; type: "insight" | "alert" | "momentum" }> {
    const insights: Array<{ message: string; timestamp?: string; type: "insight" | "alert" | "momentum" }> = [];
    
    // EPA-based insight
    if (game.home_stats.epa_per_play > 0.1) {
        insights.push({
            message: `${game.home_team} showing strong offensive efficiency (+${game.home_stats.epa_per_play.toFixed(2)} EPA/play)`,
            type: "insight"
        });
    } else if (game.away_stats.epa_per_play > 0.1) {
        insights.push({
            message: `${game.away_team} showing strong offensive efficiency (+${game.away_stats.epa_per_play.toFixed(2)} EPA/play)`,
            type: "insight"
        });
    }
    
    // Turnover insight
    if (game.home_stats.turnovers > game.away_stats.turnovers) {
        insights.push({
            message: `Turnover differential favors ${game.away_team} (${game.home_stats.turnovers} - ${game.away_stats.turnovers})`,
            type: "momentum"
        });
    } else if (game.away_stats.turnovers > game.home_stats.turnovers) {
        insights.push({
            message: `Turnover differential favors ${game.home_team} (${game.away_stats.turnovers} - ${game.home_stats.turnovers})`,
            type: "momentum"
        });
    }
    
    // Model vs market insight
    if (prediction && market) {
        const edge = Math.abs(prediction.edge_spread);
        if (edge >= 2) {
            insights.push({
                message: `Significant market inefficiency detected: ${edge.toFixed(1)} pts of model edge`,
                type: "alert"
            });
        }
    }
    
    // Default insight if none generated
    if (insights.length === 0) {
        insights.push({
            message: "Model analysis in progress. Check back for live updates.",
            type: "insight"
        });
    }
    
    return insights.slice(0, 3); // Max 3 insights
}

function generateAlerts(
    game: GameDetail,
    prediction: GameDetail["prediction"],
    market: GameDetail["market"]
): Array<{ message: string; severity: "high" | "medium" | "low" }> {
    const alerts: Array<{ message: string; severity: "high" | "medium" | "low" }> = [];
    
    // Strong edge alert
    if (prediction && Math.abs(prediction.edge_spread) >= 3) {
        alerts.push({
            message: `Strong edge detected: Model line ${prediction.edge_spread > 0 ? "above" : "below"} market by ${Math.abs(prediction.edge_spread).toFixed(1)} pts`,
            severity: "high"
        });
    }
    
    // High confidence game
    if (prediction && prediction.confidence_score >= 85) {
        alerts.push({
            message: `High confidence game (${prediction.confidence_score}%) - Model signals strong conviction`,
            severity: "medium"
        });
    }
    
    // Total line discrepancy
    if (prediction && market && Math.abs(prediction.edge_total) >= 2) {
        alerts.push({
            message: `Total line opportunity: Model projects ${prediction.predicted_total} vs market ${market.total}`,
            severity: "medium"
        });
    }
    
    return alerts;
}

export default AIIntelligenceRail;

