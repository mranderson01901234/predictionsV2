"use client";

import { GameDetail } from "@/lib/mock_data";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
import { Zap, Brain, Calendar, Clock } from "lucide-react";

interface HeroGameCardProps {
    game: GameDetail;
}

export function HeroGameCard({ game }: HeroGameCardProps) {
    const isLive = game.status === "Live";
    const isFinal = game.status === "Final";
    
    // Get prediction data
    const prediction = game.prediction;
    
    // Calculate which team model favors
    const modelFavorsHome = prediction ? prediction.predicted_spread < 0 : false;
    const favoredTeam = modelFavorsHome ? game.home_team : game.away_team;
    const modelEdge = prediction?.edge_spread || 0;
    const winProb = prediction?.win_prob_home || 0.5;
    const displayWinProb = modelFavorsHome ? winProb : (1 - winProb);
    const confidence = prediction?.confidence_score || 50;
    
    // Edge classification
    const getEdgeCategory = (edge: number) => {
        const absEdge = Math.abs(edge);
        if (absEdge >= 3) return { label: "Strong Edge", variant: "green" as const };
        if (absEdge >= 1.5) return { label: "Moderate Edge", variant: "blue" as const };
        return { label: "Small Edge", variant: "muted" as const };
    };
    
    const edgeInfo = getEdgeCategory(modelEdge);

    return (
        <motion.div
            className="glass-surface-hero p-5 lg:p-6"
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
        >
            <div className="flex flex-col lg:flex-row lg:items-center gap-4 lg:gap-6">
                
                {/* Left: Status & Game Info */}
                <div className="flex flex-col gap-2 lg:w-[140px] flex-shrink-0">
                    {/* Status Badge */}
                    {isLive && (
                        <div className="flex items-center gap-2 neon-badge-green w-fit">
                            <span className="live-indicator" />
                            <span>LIVE</span>
                        </div>
                    )}
                    {isFinal && (
                        <span className="neon-badge neon-badge-purple w-fit">FINAL</span>
                    )}
                    {!isLive && !isFinal && (
                        <span className="neon-badge text-[var(--muted-foreground)] bg-[var(--glass-bg-elevated)] border border-[var(--glass-border)] w-fit">
                            {new Date(game.date).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}
                        </span>
                    )}
                    
                    {/* Quarter & Time for Live games */}
                    {isLive && (
                        <div className="flex items-center gap-1.5 text-[var(--foreground)]">
                            <Clock size={12} className="text-[var(--muted-foreground)]" />
                            <span className="text-sm font-mono font-semibold">
                                Q{game.quarter} · {game.time_remaining}
                            </span>
                        </div>
                    )}
                    
                    {/* Week & Date */}
                    <div className="flex items-center gap-1.5 text-[var(--muted-foreground)]">
                        <Calendar size={12} />
                        <span className="text-[11px]">
                            Week {game.week} · {new Date(game.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                        </span>
                    </div>
                </div>

                {/* Center: Teams & Score - ENLARGED & HORIZONTAL */}
                <div className="flex-1 flex items-center justify-center gap-3 lg:gap-4">
                    {/* Away Team */}
                    <div className="flex items-center gap-3">
                        <div className="relative">
                            {game.away_logo ? (
                                <img 
                                    src={game.away_logo} 
                                    alt={game.away_team} 
                                    className={cn(
                                        "w-12 h-12 lg:w-14 lg:h-14 object-contain transition-opacity",
                                        game.away_score < game.home_score && "opacity-50"
                                    )} 
                                />
                            ) : (
                                <div className={cn(
                                    "w-12 h-12 lg:w-14 lg:h-14 rounded-full bg-[var(--glass-bg-elevated)] border border-[var(--glass-border)] flex items-center justify-center text-lg font-bold",
                                    game.away_score >= game.home_score ? "text-[var(--foreground)]" : "text-[var(--muted-foreground)]"
                                )}>
                                    {game.away_team[0]}
                                </div>
                            )}
                            {game.possession === 'away' && isLive && (
                                <div className="absolute -bottom-0.5 -right-0.5 w-2.5 h-2.5 bg-[var(--neon-green)] rounded-full border-2 border-[var(--background)]" />
                            )}
                        </div>
                        <div className="text-right">
                            <div className={cn(
                                "font-bold text-sm lg:text-base",
                                game.away_score >= game.home_score ? "text-[var(--foreground)]" : "text-[var(--muted-foreground)]"
                            )}>
                                {game.away_team}
                            </div>
                            <div className="text-[10px] text-[var(--muted-foreground)]">
                                {game.away_record}
                            </div>
                        </div>
                    </div>

                    {/* Score Display */}
                    <div className="flex items-center gap-3 lg:gap-4 px-2 lg:px-4">
                        <span className={cn(
                            "hero-number text-4xl lg:text-5xl tabular-nums",
                            game.away_score > game.home_score 
                                ? "text-[var(--foreground)]" 
                                : "text-[var(--muted-foreground)]"
                        )}>
                            {game.away_score}
                        </span>
                        
                        <span className="text-lg text-[var(--muted-foreground)]/50 font-light">–</span>
                        
                        <span className={cn(
                            "hero-number text-4xl lg:text-5xl tabular-nums",
                            game.home_score > game.away_score 
                                ? "text-[var(--foreground)]" 
                                : "text-[var(--muted-foreground)]"
                        )}>
                            {game.home_score}
                        </span>
                    </div>

                    {/* Home Team */}
                    <div className="flex items-center gap-3">
                        <div className="text-left">
                            <div className={cn(
                                "font-bold text-sm lg:text-base",
                                game.home_score >= game.away_score ? "text-[var(--foreground)]" : "text-[var(--muted-foreground)]"
                            )}>
                                {game.home_team}
                            </div>
                            <div className="text-[10px] text-[var(--muted-foreground)]">
                                {game.home_record}
                            </div>
                        </div>
                        <div className="relative">
                            {game.home_logo ? (
                                <img 
                                    src={game.home_logo} 
                                    alt={game.home_team} 
                                    className={cn(
                                        "w-12 h-12 lg:w-14 lg:h-14 object-contain transition-opacity",
                                        game.home_score < game.away_score && "opacity-50"
                                    )} 
                                />
                            ) : (
                                <div className={cn(
                                    "w-12 h-12 lg:w-14 lg:h-14 rounded-full bg-[var(--glass-bg-elevated)] border border-[var(--glass-border)] flex items-center justify-center text-lg font-bold",
                                    game.home_score >= game.away_score ? "text-[var(--foreground)]" : "text-[var(--muted-foreground)]"
                                )}>
                                    {game.home_team[0]}
                                </div>
                            )}
                            {game.possession === 'home' && isLive && (
                                <div className="absolute -bottom-0.5 -left-0.5 w-2.5 h-2.5 bg-[var(--neon-green)] rounded-full border-2 border-[var(--background)]" />
                            )}
                        </div>
                    </div>
                </div>

                {/* Right: Model Prediction Summary */}
                <div className="lg:w-[280px] flex-shrink-0 lg:border-l lg:border-[var(--glass-border)] lg:pl-6">
                    {/* Model Prediction Header */}
                    <div className="flex items-center gap-2 mb-3">
                        <div className="p-1.5 rounded-lg bg-[var(--glass-bg-elevated)] border border-[var(--glass-border)]">
                            <Brain size={12} className="text-[var(--muted-foreground)]" />
                        </div>
                        <span className="text-[10px] uppercase tracking-wider text-[var(--muted-foreground)] font-medium">
                            Model Prediction
                        </span>
                    </div>

                    {/* Key Metrics Row */}
                    <div className="flex items-center gap-5 mb-3">
                        {/* Win Probability */}
                        <div className="flex flex-col">
                            <span className="text-[var(--foreground)] hero-number text-2xl">
                                {Math.round(displayWinProb * 100)}%
                            </span>
                            <span className="text-[9px] text-[var(--muted-foreground)] font-medium">
                                {favoredTeam} Win Prob
                            </span>
                        </div>

                        {/* Model Edge */}
                        <div className="flex flex-col">
                            <span className={cn(
                                "hero-number text-2xl",
                                modelEdge > 0 ? "text-[var(--success)]" : modelEdge < 0 ? "text-[var(--destructive)]" : "text-[var(--foreground)]"
                            )}>
                                {modelEdge > 0 ? "+" : ""}{Math.abs(modelEdge).toFixed(1)}
                            </span>
                            <span className="text-[9px] text-[var(--muted-foreground)] font-medium">
                                Model Edge
                            </span>
                        </div>

                        {/* Confidence */}
                        <div className="flex flex-col">
                            <span className="text-[var(--foreground)] hero-number text-2xl">
                                {confidence}
                            </span>
                            <span className="text-[9px] text-[var(--muted-foreground)] font-medium">
                                Confidence
                            </span>
                        </div>
                    </div>

                    {/* AI Summary Line */}
                    <div className="glass-inner p-2.5 flex items-start gap-2">
                        <Zap size={12} className="text-[var(--muted-foreground)] mt-0.5 flex-shrink-0" />
                        <p className="text-[11px] text-[var(--muted-foreground)] leading-relaxed">
                            <span className="text-[var(--foreground)] font-medium">Model favors {favoredTeam}</span>
                            {" by "}
                            <span className="text-[var(--foreground)] font-semibold">
                                {Math.abs(prediction?.predicted_spread || 0).toFixed(1)}
                            </span>
                            {" pts "}
                            <span className={cn(
                                "neon-badge ml-1 text-[8px]",
                                edgeInfo.variant === "green" && "neon-badge-green",
                                edgeInfo.variant === "blue" && "neon-badge-blue",
                                edgeInfo.variant === "muted" && "bg-[var(--glass-bg-elevated)] text-[var(--muted-foreground)] border-[var(--glass-border)]"
                            )}>
                                {edgeInfo.label}
                            </span>
                        </p>
                    </div>
                </div>
            </div>
        </motion.div>
    );
}

export default HeroGameCard;
