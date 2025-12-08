"use client";

import { Game } from "@/lib/mock_data";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
import { useRef } from "react";

interface LiveScoreboardProps {
    games: Game[];
    selectedGameId: string;
    onSelect: (gameId: string) => void;
}

export function LiveScoreboard({ games, selectedGameId, onSelect }: LiveScoreboardProps) {
    const scrollContainerRef = useRef<HTMLDivElement>(null);

    return (
        <div className="w-full border-b border-[#2a2a2a] bg-[#0a0a0a]">
            <div
                ref={scrollContainerRef}
                className="flex overflow-x-auto py-3 px-4 gap-3 scrollbar-hide snap-x"
            >
                {games.map((game) => {
                    const isSelected = game.game_id === selectedGameId;
                    const isLive = game.status === "Live";

                    // Format time if scheduled
                    const timeString = new Date(game.date).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });

                    return (
                        <motion.button
                            key={game.game_id}
                            onClick={() => onSelect(game.game_id)}
                            className={cn(
                                "flex flex-col min-w-[140px] p-3 rounded-xl border transition-all duration-200 text-left relative overflow-hidden snap-start",
                                "bg-[#1a1a1a] border-[#2a2a2a]", // bg-card, border-subtle
                                isSelected
                                    ? "border-[#3b82f6] shadow-[0_0_15px_rgba(59,130,246,0.1)]" // border-accent-blue
                                    : "hover:bg-[#242424]" // bg-hover
                            )}
                            whileHover={{ scale: 1.02 }}
                            whileTap={{ scale: 0.98 }}
                        >
                            {/* Status Indicator */}
                            <div className="flex justify-between items-center mb-3">
                                <span className={cn(
                                    "text-[10px] font-medium uppercase tracking-wider",
                                    isLive ? "text-[#22c55e]" : "text-[#666666]" // accent-green vs text-muted
                                )}>
                                    {isLive ? "Live" : game.status === "Final" ? "Final" : timeString}
                                </span>
                                {isLive && (
                                    <span className="relative flex h-2 w-2">
                                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#22c55e] opacity-75"></span>
                                        <span className="relative inline-flex rounded-full h-2 w-2 bg-[#22c55e]"></span>
                                    </span>
                                )}
                            </div>

                            {/* Teams & Scores */}
                            <div className="space-y-2">
                                {/* Away Team */}
                                <div className="flex justify-between items-center">
                                    <div className="flex items-center gap-2">
                                        {game.away_logo ? (
                                            <img src={game.away_logo} alt={game.away_team} className="w-6 h-6 object-contain" />
                                        ) : (
                                            <div className="w-6 h-6 rounded-full bg-zinc-800 flex items-center justify-center text-[10px] text-zinc-400">
                                                {game.away_team[0]}
                                            </div>
                                        )}
                                        <div className="flex flex-col leading-none">
                                            <span className="font-bold text-sm text-[#ffffff]">{game.away_team}</span>
                                            <span className="text-[10px] text-[#666666] mt-0.5">{game.away_record || "0-0"}</span>
                                        </div>
                                    </div>
                                    <span className="font-mono text-sm font-semibold text-[#ffffff]">{game.away_score}</span>
                                </div>

                                {/* Home Team */}
                                <div className="flex justify-between items-center">
                                    <div className="flex items-center gap-2">
                                        {game.home_logo ? (
                                            <img src={game.home_logo} alt={game.home_team} className="w-6 h-6 object-contain" />
                                        ) : (
                                            <div className="w-6 h-6 rounded-full bg-zinc-800 flex items-center justify-center text-[10px] text-zinc-400">
                                                {game.home_team[0]}
                                            </div>
                                        )}
                                        <div className="flex flex-col leading-none">
                                            <span className="font-bold text-sm text-[#ffffff]">{game.home_team}</span>
                                            <span className="text-[10px] text-[#666666] mt-0.5">{game.home_record || "0-0"}</span>
                                        </div>
                                    </div>
                                    <span className="font-mono text-sm font-semibold text-[#ffffff]">{game.home_score}</span>
                                </div>
                            </div>
                        </motion.button>
                    );
                })}
            </div>
        </div>
    );
}
