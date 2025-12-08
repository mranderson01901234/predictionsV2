"use client";

import { Game, MarketSnapshot, Prediction } from "@/lib/mock_data";
import { GlassCard } from "@/components/ui/glass";
import { cn } from "@/lib/utils";
import { ArrowRight, TrendingUp, AlertTriangle, CheckCircle2 } from "lucide-react";
import Link from "next/link";

interface GameRowProps {
    game: Game;
    market?: MarketSnapshot;
    prediction?: Prediction;
}

export function GameRow({ game, market, prediction }: GameRowProps) {
    const isLive = game.status === "Live";
    const hasEdge = prediction && Math.abs(prediction.edge_spread) > 1.0;

    return (
        <Link href={`/games/${game.game_id}`}>
            <GlassCard hoverEffect className="mb-4 group relative overflow-hidden">
                {/* Edge Indicator Strip */}
                {hasEdge && (
                    <div className="absolute left-0 top-0 bottom-0 w-1 bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.5)]" />
                )}

                <div className="flex flex-col md:flex-row items-center justify-between gap-4">

                    {/* Game Info & Teams */}
                    <div className="flex-1 w-full">
                        <div className="flex items-center justify-between mb-2 md:mb-0">
                            <div className="flex items-center gap-2 text-xs font-mono text-slate-400">
                                {isLive ? (
                                    <span className="flex items-center gap-1 text-rose-400 animate-pulse">
                                        <span className="w-2 h-2 rounded-full bg-rose-500" />
                                        LIVE • {game.quarter}Q {game.time_remaining}
                                    </span>
                                ) : (
                                    <span>{new Date(game.date).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                                )}
                                <span className="hidden md:inline">• Week {game.week}</span>
                            </div>
                        </div>

                        <div className="flex items-center justify-between gap-8 mt-2">
                            {/* Away Team */}
                            <div className="flex items-center gap-3 flex-1 justify-end">
                                <span className={cn("text-lg font-bold", game.possession === 'away' && "text-emerald-400")}>
                                    {game.away_team}
                                </span>
                                <span className="text-2xl font-mono">{game.away_score}</span>
                            </div>

                            <div className="text-slate-600 text-sm font-mono">@</div>

                            {/* Home Team */}
                            <div className="flex items-center gap-3 flex-1">
                                <span className="text-2xl font-mono">{game.home_score}</span>
                                <span className={cn("text-lg font-bold", game.possession === 'home' && "text-emerald-400")}>
                                    {game.home_team}
                                </span>
                            </div>
                        </div>
                    </div>

                    {/* Market & Model Data */}
                    <div className="flex items-center gap-4 w-full md:w-auto justify-between md:justify-end border-t md:border-t-0 border-slate-800 pt-4 md:pt-0">

                        {/* Market Line */}
                        <div className="flex flex-col items-center px-4 border-r border-slate-800/50">
                            <span className="text-xs text-slate-500 uppercase tracking-wider">Market</span>
                            <div className="font-mono text-slate-300">
                                {market ? (
                                    <>
                                        {market.spread_home > 0 ? "+" : ""}{market.spread_home}
                                        <span className="text-slate-600 mx-1">/</span>
                                        {market.total}
                                    </>
                                ) : (
                                    "--"
                                )}
                            </div>
                        </div>

                        {/* Model Prediction */}
                        <div className="flex flex-col items-center px-4">
                            <span className="text-xs text-slate-500 uppercase tracking-wider flex items-center gap-1">
                                Model
                                {hasEdge && <TrendingUp className="w-3 h-3 text-emerald-400" />}
                            </span>
                            <div className={cn("font-mono font-bold", hasEdge ? "text-emerald-400" : "text-blue-400")}>
                                {prediction ? (
                                    <>
                                        {prediction.predicted_spread > 0 ? "+" : ""}{prediction.predicted_spread.toFixed(1)}
                                    </>
                                ) : (
                                    "--"
                                )}
                            </div>
                        </div>

                        {/* Action Button */}
                        <div className="pl-2">
                            <div className="p-2 rounded-full bg-slate-800 text-slate-400 group-hover:bg-emerald-500/20 group-hover:text-emerald-400 transition-colors">
                                <ArrowRight className="w-5 h-5" />
                            </div>
                        </div>
                    </div>

                </div>
            </GlassCard>
        </Link>
    );
}
