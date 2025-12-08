"use client";

import { Game } from "@/lib/mock_data";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
import { GameTileSkeleton } from "./DashboardCard";

interface GameStripProps {
    games: Game[];
    selectedGameId: string;
    onSelect: (gameId: string) => void;
    loading?: boolean;
}

interface GameTileProps {
    game: Game;
    isSelected: boolean;
    onSelect: () => void;
}

function GameTile({ game, isSelected, onSelect }: GameTileProps) {
    const isLive = game.status === "Live";
    const isFinal = game.status === "Final";

    // Format time if scheduled
    const timeString = new Date(game.date).toLocaleTimeString([], {
        hour: "numeric",
        minute: "2-digit",
    });

    return (
        <motion.button
            onClick={onSelect}
            className={cn(
                "relative flex flex-col w-full p-4 rounded-xl border transition-all duration-200 text-left",
                "bg-white/[0.02]",
                isSelected
                    ? "border-emerald-500/50 bg-emerald-500/5"
                    : "border-white/5 hover:border-white/10 hover:bg-white/[0.03]"
            )}
            whileHover={{ x: 2 }}
            whileTap={{ scale: 0.98 }}
            transition={{ duration: 0.15 }}
        >
            {/* Selected indicator bar */}
            {isSelected && (
                <motion.div
                    className="absolute left-0 top-4 bottom-4 w-0.5 bg-emerald-400 rounded-full"
                    layoutId="selectedGameBar"
                    transition={{ type: "spring", bounce: 0.2, duration: 0.4 }}
                />
            )}

            {/* Status Badge */}
            <div className="flex justify-between items-center mb-3">
                <span
                    className={cn(
                        "text-[10px] font-semibold uppercase tracking-wider px-2 py-1 rounded flex items-center gap-1",
                        isLive && "bg-emerald-500/10 text-emerald-400",
                        isFinal && "bg-purple-500/10 text-purple-400",
                        !isLive && !isFinal && "text-white/40 bg-white/5"
                    )}
                >
                    {isLive && (
                        <span className="relative flex h-2 w-2">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-current opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-2 w-2 bg-current"></span>
                        </span>
                    )}
                    {isLive ? "Live" : isFinal ? "Final" : timeString}
                </span>

                {/* Quarter indicator for live games */}
                {isLive && game.quarter && (
                    <span className="text-[10px] font-mono text-white/40">Q{game.quarter}</span>
                )}
            </div>

            {/* Teams & Scores */}
            <div className="space-y-2.5">
                {/* Away Team */}
                <div className="flex justify-between items-center">
                    <div className="flex items-center gap-2.5">
                        {game.away_logo ? (
                            <img
                                src={game.away_logo}
                                alt={game.away_team}
                                className={cn(
                                    "w-6 h-6 object-contain transition-opacity",
                                    game.away_score > game.home_score ? "opacity-100" : "opacity-50"
                                )}
                            />
                        ) : (
                            <div
                                className={cn(
                                    "w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-bold",
                                    game.away_score > game.home_score
                                        ? "bg-white/10 text-white"
                                        : "bg-white/5 text-white/40"
                                )}
                            >
                                {game.away_team[0]}
                            </div>
                        )}
                        <span
                            className={cn(
                                "font-medium text-sm",
                                game.away_score > game.home_score ? "text-white" : "text-white/50"
                            )}
                        >
                            {game.away_team}
                        </span>
                    </div>
                    <span
                        className={cn(
                            "font-mono text-sm font-bold tabular-nums",
                            game.away_score > game.home_score ? "text-white" : "text-white/50"
                        )}
                    >
                        {game.away_score}
                    </span>
                </div>

                {/* Home Team */}
                <div className="flex justify-between items-center">
                    <div className="flex items-center gap-2.5">
                        {game.home_logo ? (
                            <img
                                src={game.home_logo}
                                alt={game.home_team}
                                className={cn(
                                    "w-6 h-6 object-contain transition-opacity",
                                    game.home_score > game.away_score ? "opacity-100" : "opacity-50"
                                )}
                            />
                        ) : (
                            <div
                                className={cn(
                                    "w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-bold",
                                    game.home_score > game.away_score
                                        ? "bg-white/10 text-white"
                                        : "bg-white/5 text-white/40"
                                )}
                            >
                                {game.home_team[0]}
                            </div>
                        )}
                        <span
                            className={cn(
                                "font-medium text-sm",
                                game.home_score > game.away_score ? "text-white" : "text-white/50"
                            )}
                        >
                            {game.home_team}
                        </span>
                    </div>
                    <span
                        className={cn(
                            "font-mono text-sm font-bold tabular-nums",
                            game.home_score > game.away_score ? "text-white" : "text-white/50"
                        )}
                    >
                        {game.home_score}
                    </span>
                </div>
            </div>
        </motion.button>
    );
}

export function GameStrip({ games, selectedGameId, onSelect, loading = false }: GameStripProps) {
    if (loading) {
        return (
            <aside className="w-[240px] flex-shrink-0 bg-[#0a0a0a] border-r border-white/5 overflow-y-auto">
                <div className="p-3 space-y-3">
                    {[...Array(8)].map((_, i) => (
                        <GameTileSkeleton key={i} />
                    ))}
                </div>
            </aside>
        );
    }

    return (
        <aside className="w-[240px] flex-shrink-0 bg-[#0a0a0a] border-r border-white/5 flex flex-col h-full">
            {/* Header */}
            <div className="p-4 border-b border-white/5">
                <span className="text-xs font-medium text-white/40 uppercase tracking-wider">
                    Games ({games.length})
                </span>
            </div>

            {/* Scrollable Game List */}
            <div className="flex-1 overflow-y-auto scrollbar-thin scrollbar-thumb-white/10 scrollbar-track-transparent">
                <div className="p-3 space-y-3">
                    {games.map((game) => (
                        <GameTile
                            key={game.game_id}
                            game={game}
                            isSelected={game.game_id === selectedGameId}
                            onSelect={() => onSelect(game.game_id)}
                        />
                    ))}
                </div>
            </div>
        </aside>
    );
}

export default GameStrip;
