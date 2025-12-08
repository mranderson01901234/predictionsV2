"use client";

import { Game } from "@/lib/mock_data";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
import { useRef, useEffect, useState } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
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
        hour: 'numeric', 
        minute: '2-digit' 
    });

    return (
        <motion.button
            onClick={onSelect}
            className={cn(
                "relative flex flex-col min-w-[160px] p-3 rounded-xl border transition-all duration-200 text-left snap-start",
                "bg-[var(--glass-bg-subtle)] backdrop-blur-sm",
                isSelected
                    ? "border-[var(--neon-blue)] ring-1 ring-[var(--neon-blue-muted)] shadow-[var(--glow-blue)]"
                    : "border-[var(--glass-border)] hover:border-[var(--glass-border-medium)] hover:bg-[var(--glass-bg-medium)]",
            )}
            whileHover={{ y: -2, scale: 1.01 }}
            whileTap={{ scale: 0.98 }}
            transition={{ duration: 0.15 }}
        >
            {/* Selected indicator glow */}
            {isSelected && (
                <motion.div 
                    className="absolute inset-0 rounded-xl bg-gradient-to-b from-[var(--neon-blue)]/5 to-transparent pointer-events-none"
                    layoutId="selectedGameGlow"
                    transition={{ type: "spring", bounce: 0.2, duration: 0.4 }}
                />
            )}
            
            {/* Bottom indicator bar */}
            {isSelected && (
                <motion.div 
                    className="absolute bottom-0 left-1/2 -translate-x-1/2 w-10 h-0.5 bg-[var(--neon-blue)] rounded-full"
                    layoutId="selectedGameIndicator"
                    transition={{ type: "spring", bounce: 0.2, duration: 0.4 }}
                />
            )}
            
            {/* Status Badge */}
            <div className="flex justify-between items-center mb-3">
                <span className={cn(
                    "text-[9px] font-semibold uppercase tracking-wider px-1.5 py-0.5 rounded flex items-center gap-1",
                    isLive && "neon-badge-green",
                    isFinal && "neon-badge-purple",
                    !isLive && !isFinal && "text-[var(--muted-foreground)] bg-[var(--glass-bg)]"
                )}>
                    {isLive && (
                        <span className="relative flex h-1.5 w-1.5">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-current opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-current"></span>
                        </span>
                    )}
                    {isLive ? "Live" : isFinal ? "Final" : timeString}
                </span>
                
                {/* Quarter indicator for live games */}
                {isLive && game.quarter && (
                    <span className="text-[9px] font-mono text-[var(--muted-foreground)]">
                        Q{game.quarter}
                    </span>
                )}
            </div>

            {/* Teams & Scores */}
            <div className="space-y-2">
                {/* Away Team */}
                <div className="flex justify-between items-center">
                    <div className="flex items-center gap-2">
                        {game.away_logo ? (
                            <img 
                                src={game.away_logo} 
                                alt={game.away_team} 
                                className={cn(
                                    "w-5 h-5 object-contain transition-opacity",
                                    game.away_score > game.home_score ? "opacity-100" : "opacity-60"
                                )} 
                            />
                        ) : (
                            <div className={cn(
                                "w-5 h-5 rounded-full flex items-center justify-center text-[9px] font-bold",
                                game.away_score > game.home_score 
                                    ? "bg-[var(--neon-purple-muted)] text-[var(--neon-purple)]"
                                    : "bg-[var(--glass-bg)] text-[var(--muted-foreground)]"
                            )}>
                                {game.away_team[0]}
                            </div>
                        )}
                        <div className="flex items-center gap-1.5">
                            <span className={cn(
                                "font-semibold text-sm",
                                game.away_score > game.home_score 
                                    ? "text-[var(--foreground)]" 
                                    : "text-[var(--muted-foreground)]"
                            )}>
                                {game.away_team}
                            </span>
                            <span className="text-[9px] text-[var(--muted-foreground)] opacity-60">
                                {game.away_record || "0-0"}
                            </span>
                        </div>
                    </div>
                    <span className={cn(
                        "font-mono text-sm font-bold tabular-nums",
                        game.away_score > game.home_score 
                            ? "text-[var(--foreground)]" 
                            : "text-[var(--muted-foreground)]"
                    )}>
                        {game.away_score}
                    </span>
                </div>

                {/* Home Team */}
                <div className="flex justify-between items-center">
                    <div className="flex items-center gap-2">
                        {game.home_logo ? (
                            <img 
                                src={game.home_logo} 
                                alt={game.home_team} 
                                className={cn(
                                    "w-5 h-5 object-contain transition-opacity",
                                    game.home_score > game.away_score ? "opacity-100" : "opacity-60"
                                )} 
                            />
                        ) : (
                            <div className={cn(
                                "w-5 h-5 rounded-full flex items-center justify-center text-[9px] font-bold",
                                game.home_score > game.away_score 
                                    ? "bg-[var(--neon-blue-muted)] text-[var(--neon-blue)]"
                                    : "bg-[var(--glass-bg)] text-[var(--muted-foreground)]"
                            )}>
                                {game.home_team[0]}
                            </div>
                        )}
                        <div className="flex items-center gap-1.5">
                            <span className={cn(
                                "font-semibold text-sm",
                                game.home_score > game.away_score 
                                    ? "text-[var(--foreground)]" 
                                    : "text-[var(--muted-foreground)]"
                            )}>
                                {game.home_team}
                            </span>
                            <span className="text-[9px] text-[var(--muted-foreground)] opacity-60">
                                {game.home_record || "0-0"}
                            </span>
                        </div>
                    </div>
                    <span className={cn(
                        "font-mono text-sm font-bold tabular-nums",
                        game.home_score > game.away_score 
                            ? "text-[var(--foreground)]" 
                            : "text-[var(--muted-foreground)]"
                    )}>
                        {game.home_score}
                    </span>
                </div>
            </div>
        </motion.button>
    );
}

export function GameStrip({ games, selectedGameId, onSelect, loading = false }: GameStripProps) {
    const scrollContainerRef = useRef<HTMLDivElement>(null);
    const [canScrollLeft, setCanScrollLeft] = useState(false);
    const [canScrollRight, setCanScrollRight] = useState(false);

    const checkScroll = () => {
        const container = scrollContainerRef.current;
        if (container) {
            setCanScrollLeft(container.scrollLeft > 0);
            setCanScrollRight(
                container.scrollLeft < container.scrollWidth - container.clientWidth - 1
            );
        }
    };

    useEffect(() => {
        checkScroll();
        const container = scrollContainerRef.current;
        if (container) {
            container.addEventListener('scroll', checkScroll);
            window.addEventListener('resize', checkScroll);
            return () => {
                container.removeEventListener('scroll', checkScroll);
                window.removeEventListener('resize', checkScroll);
            };
        }
    }, [games]);

    const scroll = (direction: 'left' | 'right') => {
        const container = scrollContainerRef.current;
        if (container) {
            const scrollAmount = 300;
            container.scrollBy({
                left: direction === 'left' ? -scrollAmount : scrollAmount,
                behavior: 'smooth'
            });
        }
    };

    if (loading) {
        return (
            <div className="w-full border-b border-[var(--glass-border)] bg-[var(--glass-bg)]">
                <div className="flex py-3 px-4 gap-3 overflow-hidden">
                    {[...Array(6)].map((_, i) => (
                        <GameTileSkeleton key={i} />
                    ))}
                </div>
            </div>
        );
    }

    return (
        <div className="relative w-full border-b border-[var(--glass-border)] bg-[var(--glass-bg)] backdrop-blur-md">
            {/* Scroll Left Button */}
            {canScrollLeft && (
                <div className="absolute left-0 top-0 bottom-0 z-10 flex items-center pl-2 bg-gradient-to-r from-[var(--background)] via-[var(--background)]/90 to-transparent w-16">
                    <Button
                        variant="ghost"
                        size="icon-sm"
                        onClick={() => scroll('left')}
                        className="bg-[var(--glass-bg-elevated)] border border-[var(--glass-border)] hover:bg-[var(--glass-bg-medium)] hover:border-[var(--glass-border-medium)] shadow-lg"
                    >
                        <ChevronLeft size={16} />
                    </Button>
                </div>
            )}

            {/* Games Container */}
            <div
                ref={scrollContainerRef}
                className="flex overflow-x-auto py-3 px-4 gap-3 scrollbar-hide snap-x snap-mandatory scroll-smooth"
            >
                {games.map((game) => (
                    <GameTile
                        key={game.game_id}
                        game={game}
                        isSelected={game.game_id === selectedGameId}
                        onSelect={() => onSelect(game.game_id)}
                    />
                ))}
            </div>

            {/* Scroll Right Button */}
            {canScrollRight && (
                <div className="absolute right-0 top-0 bottom-0 z-10 flex items-center pr-2 bg-gradient-to-l from-[var(--background)] via-[var(--background)]/90 to-transparent w-16">
                    <Button
                        variant="ghost"
                        size="icon-sm"
                        onClick={() => scroll('right')}
                        className="bg-[var(--glass-bg-elevated)] border border-[var(--glass-border)] hover:bg-[var(--glass-bg-medium)] hover:border-[var(--glass-border-medium)] shadow-lg ml-auto"
                    >
                        <ChevronRight size={16} />
                    </Button>
                </div>
            )}
        </div>
    );
}

export default GameStrip;
