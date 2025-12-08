"use client";

import { GameDetail } from "@/lib/mock_data";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
import { Scale, TrendingUp, TrendingDown, Target } from "lucide-react";

interface PreGameLineCardProps {
    game: GameDetail;
}

export function PreGameLineCard({ game }: PreGameLineCardProps) {
    const { market, prediction } = game;
    
    if (!market || !prediction) {
        return (
            <motion.div 
                className="glass-surface h-full p-4"
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4 }}
            >
                <div className="flex items-center gap-2 mb-3">
                    <Scale size={14} className="text-[var(--muted-foreground)]" />
                    <span className="text-xs font-medium text-[var(--muted-foreground)] uppercase tracking-wide">
                        Market vs Model
                    </span>
                </div>
                <div className="text-center text-sm text-[var(--muted-foreground)] py-4">
                    Line data unavailable
                </div>
            </motion.div>
        );
    }

    const isHomeFavored = market.spread_home < 0;
    const favoredTeam = isHomeFavored ? game.home_team : game.away_team;
    
    // Format spreads
    const formatSpread = (spread?: number) => {
        if (spread === undefined) return "--";
        if (spread === 0) return "PK";
        return spread > 0 ? `+${spread}` : `${spread}`;
    };

    // Calculate edge magnitude for badge
    const absEdge = Math.abs(prediction.edge_spread);
    const getEdgeBadge = () => {
        if (absEdge >= 3) return { label: "Strong Edge", variant: "green" as const };
        if (absEdge >= 1.5) return { label: "Moderate", variant: "blue" as const };
        if (absEdge >= 0.5) return { label: "Small Edge", variant: "purple" as const };
        return { label: "Neutral", variant: "muted" as const };
    };
    
    const edgeInfo = getEdgeBadge();
    const hasSignificantEdge = absEdge >= 1.0;

    return (
        <motion.div 
            className={cn(
                "h-full p-4 transition-all duration-300",
                hasSignificantEdge ? "glass-card-neon" : "glass-surface"
            )}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
        >
            {/* Header with icon */}
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                    <div className={cn(
                        "p-1.5 rounded-lg border",
                        hasSignificantEdge 
                            ? "bg-[var(--neon-blue-muted)] border-[var(--neon-blue)]/20" 
                            : "bg-[var(--glass-bg-elevated)] border-[var(--glass-border)]"
                    )}>
                        <Scale size={12} className={cn(
                            hasSignificantEdge ? "text-[var(--neon-blue)]" : "text-[var(--muted-foreground)]"
                        )} />
                    </div>
                    <span className="text-xs font-semibold text-[var(--foreground)] uppercase tracking-wide">
                        Market vs Model
                    </span>
                </div>
                
                {/* Edge Badge */}
                <span className={cn(
                    "neon-badge text-[9px]",
                    edgeInfo.variant === "green" && "neon-badge-green",
                    edgeInfo.variant === "blue" && "neon-badge-blue",
                    edgeInfo.variant === "purple" && "neon-badge-purple",
                    edgeInfo.variant === "muted" && "bg-[var(--glass-bg-elevated)] text-[var(--muted-foreground)] border-[var(--glass-border)]"
                )}>
                    {edgeInfo.label}
                </span>
            </div>

            {/* Spread Comparison Row */}
            <div className="grid grid-cols-3 gap-2 mb-4">
                {/* Opening Line */}
                <div className="glass-inner p-2.5 text-center">
                    <div className="text-[9px] text-[var(--muted-foreground)] uppercase font-medium tracking-wide mb-1">
                        Open
                    </div>
                    <div className="text-base font-bold font-mono text-[var(--muted-foreground)]">
                        {formatSpread(market.spread_home_open)}
                    </div>
                </div>

                {/* Current Market Line */}
                <div className="glass-inner p-2.5 text-center">
                    <div className="text-[9px] text-[var(--muted-foreground)] uppercase font-medium tracking-wide mb-1">
                        Market
                    </div>
                    <div className="text-base font-bold font-mono text-[var(--foreground)]">
                        {formatSpread(market.spread_home)}
                    </div>
                </div>

                {/* Model Prediction */}
                <div className={cn(
                    "p-2.5 text-center rounded-lg border transition-all",
                    hasSignificantEdge 
                        ? "bg-[var(--neon-blue-muted)] border-[var(--neon-blue)]/30" 
                        : "glass-inner"
                )}>
                    <div className="text-[9px] text-[var(--muted-foreground)] uppercase font-medium tracking-wide mb-1">
                        Model
                    </div>
                    <div className={cn(
                        "text-base font-bold font-mono",
                        hasSignificantEdge ? "text-neon-blue" : "text-[var(--foreground)]"
                    )}>
                        {formatSpread(Number(prediction.predicted_spread.toFixed(1)))}
                    </div>
                </div>
            </div>

            {/* Edge Indicator Bar */}
            <div className="space-y-2">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-1.5">
                        <Target size={10} className="text-[var(--muted-foreground)]" />
                        <span className="text-[10px] text-[var(--muted-foreground)] uppercase tracking-wide font-medium">
                            Model Edge
                        </span>
                    </div>
                    <div className="flex items-center gap-1.5">
                        {prediction.edge_spread > 0 ? (
                            <TrendingUp size={12} className="text-[var(--neon-green)]" />
                        ) : prediction.edge_spread < 0 ? (
                            <TrendingDown size={12} className="text-[var(--destructive)]" />
                        ) : null}
                        <span className={cn(
                            "text-sm font-mono font-bold",
                            prediction.edge_spread > 0 ? "text-[var(--neon-green)]" : 
                            prediction.edge_spread < 0 ? "text-[var(--destructive)]" : 
                            "text-[var(--foreground)]"
                        )}>
                            {prediction.edge_spread > 0 ? "+" : ""}
                            {prediction.edge_spread.toFixed(1)}
                        </span>
                    </div>
                </div>
                
                {/* Visual edge bar */}
                <div className="edge-bar h-1.5 rounded-full">
                    {/* Center line marker */}
                    <div className="absolute top-1/2 left-1/2 -translate-y-1/2 -translate-x-1/2 w-0.5 h-3 bg-[var(--glass-border-strong)] rounded z-10" />
                    
                    {/* Edge fill */}
                    {prediction.edge_spread !== 0 && (
                        <motion.div 
                            className={cn(
                                "absolute h-full rounded",
                                prediction.edge_spread > 0 
                                    ? "left-1/2 bg-gradient-to-r from-[var(--neon-green)]/60 to-[var(--neon-green)]" 
                                    : "right-1/2 bg-gradient-to-l from-[var(--destructive)]/60 to-[var(--destructive)]"
                            )}
                            initial={{ width: 0 }}
                            animate={{ 
                                width: `${Math.min(Math.abs(prediction.edge_spread) / 10 * 50, 50)}%` 
                            }}
                            transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
                            style={{
                                boxShadow: prediction.edge_spread > 0 
                                    ? '0 0 8px var(--neon-green-glow)' 
                                    : '0 0 8px rgba(255, 59, 92, 0.4)'
                            }}
                        />
                    )}
                </div>
            </div>

            {/* Total Line (condensed) */}
            <div className="mt-4 pt-3 border-t border-[var(--glass-border)] flex items-center justify-between">
                <span className="text-[10px] text-[var(--muted-foreground)] uppercase tracking-wide font-medium">
                    Total O/U
                </span>
                <div className="flex items-center gap-3 text-xs font-mono">
                    <span className="text-[var(--muted-foreground)]">
                        Mkt: <span className="text-[var(--foreground)] font-semibold">{market.total}</span>
                    </span>
                    <span className="text-[var(--muted-foreground)]">
                        Model: <span className={cn(
                            "font-semibold",
                            Math.abs(prediction.edge_total) >= 1.5 ? "text-[var(--neon-blue)]" : "text-[var(--foreground)]"
                        )}>
                            {prediction.predicted_total}
                        </span>
                    </span>
                </div>
            </div>
        </motion.div>
    );
}

export default PreGameLineCard;
