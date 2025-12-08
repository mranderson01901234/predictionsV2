"use client";

import { Game, GameDetail } from "@/lib/mock_data";
import { cn } from "@/lib/utils";

interface BettingCardProps {
    games: Game[];
    details: Record<string, GameDetail>;
}

export function BettingCard({ games, details }: BettingCardProps) {
    // Filter for games with edge
    const bets = games.map(game => {
        const detail = details[game.game_id];
        const prediction = detail?.prediction;
        const market = detail?.market;

        if (!prediction || !market) return null;

        const edge = prediction.edge_spread;
        if (Math.abs(edge) < 1.0) return null; // Only show bets with > 1.0 edge

        const isHomeBet = edge > 0;
        const team = isHomeBet ? game.home_team : game.away_team;
        const logo = isHomeBet ? game.home_logo : game.away_logo;
        const line = isHomeBet ? market.spread_home : -market.spread_home; // Invert for away
        const modelLine = isHomeBet ? prediction.predicted_spread : -prediction.predicted_spread;

        // Simple Kelly Calc (Mock)
        const winProb = 0.525 + (Math.abs(edge) * 0.02); // Mock prob based on edge
        const kelly = ((winProb * 0.91) - (1 - winProb)) / 0.91; // Standard -110 odds
        const units = Math.max(0, Math.min(5, kelly * 10)).toFixed(1); // Cap at 5u

        return {
            ...game,
            betTeam: team,
            betLogo: logo,
            line,
            modelLine,
            edge: Math.abs(edge),
            units,
            ev: (Math.abs(edge) * 2.5).toFixed(1) + "%" // Mock EV
        };
    }).filter(Boolean) as any[];

    return (
        <div className="max-w-5xl mx-auto p-6">
            <div className="mb-8">
                <h1 className="text-2xl font-bold text-white mb-2">Betting Card</h1>
                <p className="text-[#a0a0a0]">Week 14, 2025 • Recommended Plays</p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {bets.map((bet) => (
                    <div key={bet.game_id} className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-xl p-5 hover:border-[#3b82f6] transition-colors group relative overflow-hidden">
                        {/* EV Badge */}
                        <div className="absolute top-0 right-0 bg-[#22c55e]/10 text-[#22c55e] text-xs font-bold px-3 py-1 rounded-bl-xl border-b border-l border-[#22c55e]/20">
                            +{bet.ev} EV
                        </div>

                        <div className="flex justify-between items-start mb-6">
                            <div className="flex items-center gap-4">
                                {bet.betLogo ? (
                                    <img src={bet.betLogo} className="w-12 h-12 object-contain" />
                                ) : (
                                    <div className="w-12 h-12 rounded-full bg-zinc-800" />
                                )}
                                <div>
                                    <h3 className="text-xl font-bold text-white">{bet.betTeam}</h3>
                                    <div className="text-sm text-[#a0a0a0]">vs {bet.betTeam === bet.home_team ? bet.away_team : bet.home_team}</div>
                                </div>
                            </div>
                            <div className="text-right mt-8">
                                <div className="text-3xl font-bold text-white font-mono">
                                    {bet.line > 0 ? `+${bet.line}` : bet.line}
                                </div>
                                <div className="text-xs text-[#666666] uppercase tracking-wider">Current Line</div>
                            </div>
                        </div>

                        <div className="grid grid-cols-3 gap-4 border-t border-[#2a2a2a] pt-4">
                            <div>
                                <div className="text-xs text-[#666666] uppercase mb-1">Model Line</div>
                                <div className="text-lg font-bold text-[#3b82f6] font-mono">
                                    {bet.modelLine > 0 ? `+${bet.modelLine.toFixed(1)}` : bet.modelLine.toFixed(1)}
                                </div>
                            </div>
                            <div>
                                <div className="text-xs text-[#666666] uppercase mb-1">Edge</div>
                                <div className="text-lg font-bold text-[#22c55e] font-mono">
                                    {bet.edge.toFixed(1)} pts
                                </div>
                            </div>
                            <div>
                                <div className="text-xs text-[#666666] uppercase mb-1">Size</div>
                                <div className="text-lg font-bold text-white font-mono flex items-center gap-1">
                                    {bet.units}u
                                    <span className="text-[10px] text-[#666666] font-normal">Rec.</span>
                                </div>
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}
