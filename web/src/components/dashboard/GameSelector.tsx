"use client";

import { Game } from "@/lib/mock_data";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";

interface GameSelectorProps {
    games: Game[];
    selectedGameId: string;
    onSelect: (gameId: string) => void;
}

export function GameSelector({ games, selectedGameId, onSelect }: GameSelectorProps) {
    return (
        <div className="w-full overflow-x-auto pb-4 scrollbar-hide">
            <div className="flex gap-3 min-w-max px-1">
                {games.map((game) => {
                    const isSelected = game.game_id === selectedGameId;
                    const isLive = game.status === "Live";

                    return (
                        <motion.button
                            key={game.game_id}
                            onClick={() => onSelect(game.game_id)}
                            className={cn(
                                "flex flex-col min-w-[140px] p-3 rounded-xl border transition-all duration-200 text-left relative overflow-hidden",
                                isSelected
                                    ? "bg-zinc-800/80 border-zinc-600/50 shadow-[0_0_15px_rgba(255,255,255,0.05)]"
                                    : "bg-zinc-900/40 border-zinc-800 hover:bg-zinc-800/60 hover:border-zinc-700"
                            )}
                            whileHover={{ scale: 1.02 }}
                            whileTap={{ scale: 0.98 }}
                        >
                            {/* Status Indicator */}
                            <div className="flex justify-between items-center mb-2">
                                <span className={cn(
                                    "text-[10px] font-bold uppercase tracking-wider",
                                    isLive ? "text-rose-400" : "text-zinc-500"
                                )}>
                                    {isLive ? "Live" : "Final"}
                                </span>
                                {isLive && <span className="w-1.5 h-1.5 rounded-full bg-rose-500 animate-pulse" />}
                            </div>

                            {/* Teams & Scores */}
                            <div className="space-y-1">
                                <div className="flex justify-between items-center">
                                    <span className={cn("font-bold text-sm", isSelected ? "text-white" : "text-zinc-300")}>
                                        {game.away_team}
                                    </span>
                                    <span className="font-mono text-sm text-zinc-400">{game.away_score}</span>
                                </div>
                                <div className="flex justify-between items-center">
                                    <span className={cn("font-bold text-sm", isSelected ? "text-white" : "text-zinc-300")}>
                                        {game.home_team}
                                    </span>
                                    <span className="font-mono text-sm text-zinc-400">{game.home_score}</span>
                                </div>
                            </div>

                            {/* Active Indicator Bar */}
                            {isSelected && (
                                <motion.div
                                    layoutId="active-game-bar"
                                    className="absolute bottom-0 left-0 right-0 h-0.5 bg-white"
                                />
                            )}
                        </motion.button>
                    );
                })}
            </div>
        </div>
    );
}
