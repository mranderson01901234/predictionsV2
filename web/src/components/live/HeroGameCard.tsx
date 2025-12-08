"use client";

import { GameDetail } from "@/lib/mock_data";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";

interface HeroGameCardProps {
    game: GameDetail;
}

export function HeroGameCard({ game }: HeroGameCardProps) {
    const isLive = game.status === "Live";
    const isFinal = game.status === "Final";

    return (
        <motion.div
            className="h-full flex items-center justify-center"
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
        >
            {/* Floating Scoreboard */}
            <div className="flex items-center gap-6">
                {/* Away Team */}
                <div className="flex items-center gap-4">
                    <div className="relative">
                        {game.away_logo ? (
                            <img
                                src={game.away_logo}
                                alt={game.away_team}
                                className={cn(
                                    "w-16 h-16 object-contain transition-opacity",
                                    game.away_score < game.home_score && "opacity-40"
                                )}
                            />
                        ) : (
                            <div className={cn(
                                "w-16 h-16 rounded-full bg-white/5 flex items-center justify-center text-2xl font-bold",
                                game.away_score >= game.home_score ? "text-white/90" : "text-white/40"
                            )}>
                                {game.away_team[0]}
                            </div>
                        )}
                        {/* Possession indicator */}
                        {game.possession === 'away' && isLive && (
                            <div className="absolute -bottom-1 -right-1 w-3 h-3 bg-[var(--neon-green)] rounded-full border-2 border-[var(--background)]" />
                        )}
                    </div>
                    <span className={cn(
                        "text-6xl tabular-nums font-bold tracking-tight",
                        game.away_score > game.home_score ? "text-[var(--foreground)]" : "text-[var(--foreground)]/40"
                    )}>
                        {game.away_score}
                    </span>
                </div>

                {/* Center - Quarter & Time */}
                <div className="flex flex-col items-center px-4">
                    {isLive && (
                        <>
                            <div className="flex items-center gap-2 mb-1">
                                <span className="w-2 h-2 rounded-full bg-[var(--neon-green)] animate-pulse" />
                                <span className="text-sm font-semibold text-[var(--foreground)]/60 uppercase tracking-wider">
                                    Q{game.quarter}
                                </span>
                            </div>
                            <span className="text-2xl font-mono font-bold text-[var(--foreground)]">
                                {game.time_remaining}
                            </span>
                        </>
                    )}
                    {isFinal && (
                        <span className="text-lg font-semibold text-[var(--foreground)]/40 uppercase tracking-widest">
                            Final
                        </span>
                    )}
                    {!isLive && !isFinal && (
                        <span className="text-lg font-mono text-[var(--foreground)]/50">
                            {new Date(game.date).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}
                        </span>
                    )}
                </div>

                {/* Home Team */}
                <div className="flex items-center gap-4">
                    <span className={cn(
                        "text-6xl tabular-nums font-bold tracking-tight",
                        game.home_score > game.away_score ? "text-[var(--foreground)]" : "text-[var(--foreground)]/40"
                    )}>
                        {game.home_score}
                    </span>
                    <div className="relative">
                        {game.home_logo ? (
                            <img
                                src={game.home_logo}
                                alt={game.home_team}
                                className={cn(
                                    "w-16 h-16 object-contain transition-opacity",
                                    game.home_score < game.away_score && "opacity-40"
                                )}
                            />
                        ) : (
                            <div className={cn(
                                "w-16 h-16 rounded-full bg-white/5 flex items-center justify-center text-2xl font-bold",
                                game.home_score >= game.away_score ? "text-white/90" : "text-white/40"
                            )}>
                                {game.home_team[0]}
                            </div>
                        )}
                        {/* Possession indicator */}
                        {game.possession === 'home' && isLive && (
                            <div className="absolute -bottom-1 -left-1 w-3 h-3 bg-[var(--neon-green)] rounded-full border-2 border-[var(--background)]" />
                        )}
                    </div>
                </div>
            </div>
        </motion.div>
    );
}

export default HeroGameCard;
