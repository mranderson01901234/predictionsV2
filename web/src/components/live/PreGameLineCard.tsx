"use client";

import { GameDetail } from "@/lib/mock_data";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";

interface PreGameLineCardProps {
    game: GameDetail;
    expanded?: boolean;
}

export function PreGameLineCard({ game, expanded = false }: PreGameLineCardProps) {
    const { market, prediction } = game;

    if (!market || !prediction) {
        return (
            <motion.div
                className={cn(
                    "glass-surface flex items-center justify-center",
                    expanded ? "h-full" : "p-4"
                )}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4 }}
            >
                <div className="text-center text-sm text-[var(--muted-foreground)] py-2">
                    Line data unavailable
                </div>
            </motion.div>
        );
    }

    // Format spreads
    const formatSpread = (spread?: number) => {
        if (spread === undefined) return "—";
        if (spread === 0) return "PK";
        return spread > 0 ? `+${spread}` : `${spread}`;
    };

    // Calculate edge magnitude
    const absEdge = Math.abs(prediction.edge_spread);
    const hasPositiveEdge = prediction.edge_spread > 0;

    if (expanded) {
        return (
            <motion.div
                className="h-full flex items-center justify-center"
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: 0.5 }}
            >
                {/* Floating compact card */}
                <div className="glass-surface px-6 py-3 flex items-center gap-8">
                    {/* Line values */}
                    <div className="flex items-center gap-6">
                        {/* Opening Line */}
                        <div className="flex flex-col items-center">
                            <span className="text-[9px] uppercase tracking-widest text-[var(--muted-foreground)]">
                                Open
                            </span>
                            <span className="text-lg font-mono font-medium text-[var(--foreground)]/40">
                                {formatSpread(market.spread_home_open)}
                            </span>
                        </div>

                        {/* Current Market Line */}
                        <div className="flex flex-col items-center">
                            <span className="text-[9px] uppercase tracking-widest text-[var(--muted-foreground)]">
                                Market
                            </span>
                            <span className="text-lg font-mono font-bold text-[var(--foreground)]">
                                {formatSpread(market.spread_home)}
                            </span>
                        </div>

                        {/* Model Prediction */}
                        <div className="flex flex-col items-center">
                            <span className="text-[9px] uppercase tracking-widest text-[var(--muted-foreground)]">
                                Model
                            </span>
                            <span className="text-lg font-mono font-bold text-[var(--neon-green)]">
                                {formatSpread(Number(prediction.predicted_spread.toFixed(1)))}
                            </span>
                        </div>
                    </div>

                    {/* Divider */}
                    <div className="w-px h-8 bg-[var(--glass-border)]" />

                    {/* Edge badge */}
                    <div className={cn(
                        "flex items-center gap-2 px-4 py-2 rounded-lg",
                        absEdge >= 1.5
                            ? hasPositiveEdge
                                ? "bg-[var(--neon-green)]/10 border border-[var(--neon-green)]/20"
                                : "bg-[var(--destructive)]/10 border border-[var(--destructive)]/20"
                            : "bg-[var(--glass-bg-elevated)] border border-[var(--glass-border)]"
                    )}>
                        <span className={cn(
                            "text-xs uppercase tracking-wider font-medium",
                            absEdge >= 1.5
                                ? hasPositiveEdge ? "text-[var(--neon-green)]/70" : "text-[var(--destructive)]/70"
                                : "text-[var(--muted-foreground)]"
                        )}>
                            Edge
                        </span>
                        <span className={cn(
                            "text-xl font-bold font-mono",
                            absEdge >= 1.5
                                ? hasPositiveEdge ? "text-[var(--neon-green)]" : "text-[var(--destructive)]"
                                : "text-[var(--foreground)]/60"
                        )}>
                            {prediction.edge_spread > 0 ? "+" : ""}
                            {prediction.edge_spread.toFixed(1)}
                        </span>
                    </div>
                </div>
            </motion.div>
        );
    }

    // Compact mode (original)
    return (
        <motion.div
            className="market-strip"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
        >
            {/* Left side: Line values */}
            <div className="flex items-center gap-6">
                {/* Opening Line */}
                <div className="market-strip-item">
                    <span className="market-strip-label">Open</span>
                    <span className="market-strip-value text-white/40">
                        {formatSpread(market.spread_home_open)}
                    </span>
                </div>

                {/* Current Market Line */}
                <div className="market-strip-item">
                    <span className="market-strip-label">Market</span>
                    <span className="market-strip-value text-white/90">
                        {formatSpread(market.spread_home)}
                    </span>
                </div>

                {/* Model Prediction */}
                <div className="market-strip-item">
                    <span className="market-strip-label">Model</span>
                    <span className="market-strip-value text-emerald-400">
                        {formatSpread(Number(prediction.predicted_spread.toFixed(1)))}
                    </span>
                </div>
            </div>

            {/* Right side: Edge (make it POP) */}
            <div className={cn(
                "flex items-center gap-3 px-4 py-2 rounded-lg",
                absEdge >= 1.5
                    ? hasPositiveEdge
                        ? "bg-emerald-500/10 border border-emerald-500/20"
                        : "bg-red-500/10 border border-red-500/20"
                    : "bg-white/5 border border-white/5"
            )}>
                <span className={cn(
                    "text-xs uppercase tracking-wider",
                    absEdge >= 1.5
                        ? hasPositiveEdge ? "text-emerald-400/70" : "text-red-400/70"
                        : "text-white/40"
                )}>
                    Edge
                </span>
                <span className={cn(
                    "text-xl font-bold font-mono",
                    absEdge >= 1.5
                        ? hasPositiveEdge ? "text-emerald-400 edge-positive" : "text-red-400 edge-negative"
                        : "text-white/60"
                )}>
                    {prediction.edge_spread > 0 ? "+" : ""}
                    {prediction.edge_spread.toFixed(1)}
                </span>
            </div>
        </motion.div>
    );
}

export default PreGameLineCard;
