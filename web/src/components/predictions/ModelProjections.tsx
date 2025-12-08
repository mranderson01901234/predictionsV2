"use client";

import { Game, GameDetail } from "@/lib/mock_data";
import { cn } from "@/lib/utils";
import { ChevronRight } from "lucide-react";

interface ModelProjectionsProps {
    games: Game[];
    details: Record<string, GameDetail>;
}

export function ModelProjections({ games, details }: ModelProjectionsProps) {
    // Group games by date
    const groupedGames = games.reduce((acc, game) => {
        const date = new Date(game.date).toLocaleDateString("en-US", { weekday: 'long', month: 'short', day: 'numeric' });
        if (!acc[date]) acc[date] = [];
        acc[date].push(game);
        return acc;
    }, {} as Record<string, Game[]>);

    return (
        <div className="max-w-5xl mx-auto p-6">
            <div className="flex justify-between items-center mb-8">
                <h1 className="text-2xl font-bold text-white">NFL Model Projections</h1>
                <div className="flex gap-2">
                    <select className="bg-[#1a1a1a] border border-[#2a2a2a] text-white text-sm rounded-lg px-3 py-2 outline-none">
                        <option>Week 14</option>
                    </select>
                    <select className="bg-[#1a1a1a] border border-[#2a2a2a] text-white text-sm rounded-lg px-3 py-2 outline-none">
                        <option>2025</option>
                    </select>
                </div>
            </div>

            <div className="space-y-8">
                {Object.entries(groupedGames).map(([date, dayGames]) => (
                    <div key={date}>
                        <div className="flex items-center gap-2 mb-4">
                            <div className="w-2 h-2 rounded-full bg-[#3b82f6]" />
                            <h2 className="text-[#a0a0a0] font-medium">{date}</h2>
                        </div>

                        <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-xl overflow-hidden">
                            {/* Header */}
                            <div className="grid grid-cols-12 gap-4 p-4 border-b border-[#2a2a2a] text-xs text-[#666666] uppercase tracking-wider font-medium">
                                <div className="col-span-5">Matchup</div>
                                <div className="col-span-2 text-center">Score</div>
                                <div className="col-span-2 text-center">Open</div>
                                <div className="col-span-1 text-center">Close</div>
                                <div className="col-span-2 text-center">Model</div>
                            </div>

                            {/* Rows */}
                            {dayGames.map((game) => {
                                const detail = details[game.game_id];
                                const prediction = detail?.prediction;
                                const market = detail?.market;

                                const isEv = prediction && Math.abs(prediction.edge_spread) > 1.5;
                                const edgeColor = prediction?.edge_spread && prediction.edge_spread > 0 ? "text-[#22c55e]" : "text-[#ef4444]";

                                return (
                                    <div key={game.game_id} className="grid grid-cols-12 gap-4 p-4 border-b border-[#2a2a2a] last:border-0 hover:bg-[#242424] transition-colors items-center group cursor-pointer">
                                        {/* Matchup */}
                                        <div className="col-span-5">
                                            <div className="flex flex-col gap-2">
                                                <div className="flex items-center justify-between">
                                                    <div className="flex items-center gap-3">
                                                        {game.away_logo ? (
                                                            <img src={game.away_logo} className="w-6 h-6 object-contain" />
                                                        ) : (
                                                            <div className="w-6 h-6 rounded-full bg-zinc-800" />
                                                        )}
                                                        <span className="text-white font-bold">{game.away_team}</span>
                                                        <span className="text-xs text-[#666666]">{game.away_record}</span>
                                                    </div>
                                                    {prediction?.edge_spread && prediction.edge_spread < 0 && isEv && (
                                                        <span className="text-[10px] bg-[#22c55e]/10 text-[#22c55e] px-1.5 py-0.5 rounded font-bold">+EV</span>
                                                    )}
                                                </div>
                                                <div className="flex items-center justify-between">
                                                    <div className="flex items-center gap-3">
                                                        {game.home_logo ? (
                                                            <img src={game.home_logo} className="w-6 h-6 object-contain" />
                                                        ) : (
                                                            <div className="w-6 h-6 rounded-full bg-zinc-800" />
                                                        )}
                                                        <span className="text-white font-bold">{game.home_team}</span>
                                                        <span className="text-xs text-[#666666]">{game.home_record}</span>
                                                    </div>
                                                    {prediction?.edge_spread && prediction.edge_spread > 0 && isEv && (
                                                        <span className="text-[10px] bg-[#22c55e]/10 text-[#22c55e] px-1.5 py-0.5 rounded font-bold">+EV</span>
                                                    )}
                                                </div>
                                            </div>
                                        </div>

                                        {/* Score */}
                                        <div className="col-span-2 flex flex-col items-center justify-center gap-2">
                                            <span className="text-white font-mono">{game.status === 'Scheduled' ? '--' : game.away_score}</span>
                                            <span className="text-white font-mono">{game.status === 'Scheduled' ? '--' : game.home_score}</span>
                                        </div>

                                        {/* Open */}
                                        <div className="col-span-2 flex flex-col items-center justify-center gap-2 text-[#a0a0a0] text-sm">
                                            <span>{market?.spread_home_open ? (market.spread_home_open > 0 ? `+${market.spread_home_open}` : market.spread_home_open) : '--'}</span>
                                        </div>

                                        {/* Close */}
                                        <div className="col-span-1 flex flex-col items-center justify-center gap-2 text-white font-medium text-sm">
                                            <span>{market?.spread_home ? (market.spread_home > 0 ? `+${market.spread_home}` : market.spread_home) : '--'}</span>
                                        </div>

                                        {/* Model */}
                                        <div className="col-span-2 flex flex-col items-center justify-center gap-1">
                                            <span className={cn("font-bold font-mono", edgeColor)}>
                                                {prediction?.predicted_spread ? (prediction.predicted_spread > 0 ? `+${prediction.predicted_spread.toFixed(1)}` : prediction.predicted_spread.toFixed(1)) : '--'}
                                            </span>
                                            {isEv && (
                                                <span className="text-[10px] text-[#666666]">
                                                    {Math.abs(prediction!.edge_spread).toFixed(1)} diff
                                                </span>
                                            )}
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}
