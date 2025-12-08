"use client";

import { GameDetail } from "@/lib/mock_data";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
import { ClipboardList, Zap } from "lucide-react";

interface ScoringSummaryCardProps {
    game: GameDetail;
}

export function ScoringSummaryCard({ game }: ScoringSummaryCardProps) {
    const quarters = ["1", "2", "3", "4"];
    
    // Determine winning team
    const awayWinning = game.away_score > game.home_score;
    const homeWinning = game.home_score > game.away_score;
    const isLive = game.status === "Live";
    const isFinal = game.status === "Final";

    return (
        <motion.div 
            className="glass-surface h-full p-4"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
        >
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                    <div className="p-1.5 rounded-lg bg-[var(--glass-bg-elevated)] border border-[var(--glass-border)]">
                        <ClipboardList size={12} className="text-[var(--muted-foreground)]" />
                    </div>
                    <span className="text-xs font-semibold text-[var(--foreground)] uppercase tracking-wide">
                        Scoring
                    </span>
                </div>
                
                {/* Status Badge */}
                <span className={cn(
                    "neon-badge text-[9px]",
                    isLive && "neon-badge-green",
                    isFinal && "neon-badge-purple",
                    !isLive && !isFinal && "bg-[var(--glass-bg-elevated)] text-[var(--muted-foreground)] border-[var(--glass-border)]"
                )}>
                    {isLive && (
                        <span className="relative flex h-1.5 w-1.5 mr-1">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-current opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-current"></span>
                        </span>
                    )}
                    {isLive ? "Live" : isFinal ? "Final" : "Pre"}
                </span>
            </div>

            {/* Score Table */}
            <div className="overflow-x-auto scrollbar-hide">
                <table className="w-full min-w-[200px]">
                    <thead>
                        <tr className="text-[9px] text-[var(--muted-foreground)] uppercase tracking-wide">
                            <th className="text-left py-2 font-medium w-auto">Team</th>
                            {quarters.map((q) => (
                                <th key={q} className="text-center w-8 font-medium">{q}</th>
                            ))}
                            <th className="text-right w-10 font-semibold">T</th>
                        </tr>
                    </thead>
                    <tbody>
                        {/* Away Team Row */}
                        <tr className="border-y border-[var(--glass-border)] group">
                            <td className="py-2.5">
                                <div className="flex items-center gap-2">
                                    {game.away_logo ? (
                                        <img 
                                            src={game.away_logo} 
                                            alt={game.away_team} 
                                            className={cn(
                                                "w-5 h-5 object-contain flex-shrink-0 transition-all",
                                                awayWinning ? "opacity-100" : "opacity-60"
                                            )} 
                                        />
                                    ) : (
                                        <div className={cn(
                                            "w-5 h-5 rounded-full flex items-center justify-center text-[9px] font-bold flex-shrink-0",
                                            awayWinning 
                                                ? "bg-[var(--neon-purple-muted)] text-[var(--neon-purple)]" 
                                                : "bg-[var(--glass-bg)] text-[var(--muted-foreground)]"
                                        )}>
                                            {game.away_team[0]}
                                        </div>
                                    )}
                                    <span className={cn(
                                        "font-semibold text-xs transition-colors",
                                        awayWinning ? "text-[var(--foreground)]" : "text-[var(--muted-foreground)]"
                                    )}>
                                        {game.away_team}
                                    </span>
                                </div>
                            </td>
                            {game.scoring_summary.away.map((score, i) => (
                                <td key={i} className="text-center text-[var(--muted-foreground)] font-mono text-xs py-2.5 tabular-nums">
                                    {score}
                                </td>
                            ))}
                            <td className={cn(
                                "text-right font-bold font-mono text-sm py-2.5 tabular-nums",
                                awayWinning ? "text-neon-purple" : "text-[var(--muted-foreground)]"
                            )}>
                                {game.away_score}
                            </td>
                        </tr>
                        
                        {/* Home Team Row */}
                        <tr className="group">
                            <td className="py-2.5">
                                <div className="flex items-center gap-2">
                                    {game.home_logo ? (
                                        <img 
                                            src={game.home_logo} 
                                            alt={game.home_team} 
                                            className={cn(
                                                "w-5 h-5 object-contain flex-shrink-0 transition-all",
                                                homeWinning ? "opacity-100" : "opacity-60"
                                            )} 
                                        />
                                    ) : (
                                        <div className={cn(
                                            "w-5 h-5 rounded-full flex items-center justify-center text-[9px] font-bold flex-shrink-0",
                                            homeWinning 
                                                ? "bg-[var(--neon-blue-muted)] text-[var(--neon-blue)]" 
                                                : "bg-[var(--glass-bg)] text-[var(--muted-foreground)]"
                                        )}>
                                            {game.home_team[0]}
                                        </div>
                                    )}
                                    <span className={cn(
                                        "font-semibold text-xs transition-colors",
                                        homeWinning ? "text-[var(--foreground)]" : "text-[var(--muted-foreground)]"
                                    )}>
                                        {game.home_team}
                                    </span>
                                </div>
                            </td>
                            {game.scoring_summary.home.map((score, i) => (
                                <td key={i} className="text-center text-[var(--muted-foreground)] font-mono text-xs py-2.5 tabular-nums">
                                    {score}
                                </td>
                            ))}
                            <td className={cn(
                                "text-right font-bold font-mono text-sm py-2.5 tabular-nums",
                                homeWinning ? "text-neon-blue" : "text-[var(--muted-foreground)]"
                            )}>
                                {game.home_score}
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>

            {/* Live game indicator */}
            {isLive && game.quarter && game.time_remaining && (
                <div className="mt-3 pt-3 border-t border-[var(--glass-border)] flex items-center justify-center gap-2">
                    <Zap size={12} className="text-[var(--neon-green)]" />
                    <span className="text-[10px] text-[var(--muted-foreground)]">
                        Q{game.quarter} · <span className="text-[var(--foreground)] font-mono font-medium">{game.time_remaining}</span>
                    </span>
                    {game.possession && (
                        <span className="text-[10px] text-[var(--muted-foreground)]">
                            · <span className={cn(
                                "font-medium",
                                game.possession === 'home' ? "text-[var(--neon-blue)]" : "text-[var(--neon-purple)]"
                            )}>
                                {game.possession === 'home' ? game.home_team : game.away_team}
                            </span> ball
                        </span>
                    )}
                </div>
            )}
        </motion.div>
    );
}

export default ScoringSummaryCard;
