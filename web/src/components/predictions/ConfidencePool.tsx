"use client";

import { Game, GameDetail } from "@/lib/mock_data";
import { cn } from "@/lib/utils";

interface ConfidencePoolProps {
    games: Game[];
    details: Record<string, GameDetail>;
}

export function ConfidencePool({ games, details }: ConfidencePoolProps) {
    // Sort games by confidence (highest win probability)
    const sortedPicks = games.map(game => {
        const detail = details[game.game_id];
        const prob = detail?.prediction?.win_prob_home || 0.5;
        const isHomeWinner = prob > 0.5;
        const winProb = isHomeWinner ? prob : 1 - prob;
        const winner = isHomeWinner ? game.home_team : game.away_team;
        const winnerLogo = isHomeWinner ? game.home_logo : game.away_logo;
        const loser = isHomeWinner ? game.away_team : game.home_team;
        const loserLogo = isHomeWinner ? game.away_logo : game.home_logo;

        return {
            ...game,
            winProb,
            winner,
            winnerLogo,
            loser,
            loserLogo,
            rank: 0 // To be assigned
        };
    }).sort((a, b) => b.winProb - a.winProb);

    // Assign ranks
    sortedPicks.forEach((pick, index) => {
        pick.rank = sortedPicks.length - index;
    });

    return (
        <div className="max-w-4xl mx-auto p-6">
            <div className="mb-8">
                <h1 className="text-2xl font-bold text-white mb-2">NFL Confidence Pool Picks</h1>
                <p className="text-[#a0a0a0]">Week 14, 2025</p>
            </div>

            <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-xl overflow-hidden">
                <div className="grid grid-cols-12 gap-4 p-4 border-b border-[#2a2a2a] text-xs text-[#666666] uppercase tracking-wider font-medium">
                    <div className="col-span-1 text-center">Rank</div>
                    <div className="col-span-6">Projected Winner</div>
                    <div className="col-span-5 text-right">Win % / 100</div>
                </div>

                {sortedPicks.map((pick) => {
                    const confidencePercent = Math.round(pick.winProb * 100);
                    let barColor = "bg-[#71717a]"; // Neutral
                    if (confidencePercent >= 70) barColor = "bg-[#22c55e]";
                    else if (confidencePercent >= 60) barColor = "bg-[#eab308]";

                    return (
                        <div key={pick.game_id} className="grid grid-cols-12 gap-4 p-4 border-b border-[#2a2a2a] last:border-0 hover:bg-[#242424] transition-colors items-center">
                            {/* Rank */}
                            <div className="col-span-1 text-center">
                                <span className="text-xl font-bold text-white">{pick.rank}</span>
                            </div>

                            {/* Matchup */}
                            <div className="col-span-6">
                                <div className="flex items-center gap-4">
                                    {/* Winner */}
                                    <div className="flex items-center gap-3">
                                        {pick.winnerLogo ? (
                                            <img src={pick.winnerLogo} className="w-8 h-8 object-contain" />
                                        ) : (
                                            <div className="w-8 h-8 rounded-full bg-zinc-800" />
                                        )}
                                        <div className="flex flex-col">
                                            <span className="font-bold text-white">{pick.winner}</span>
                                            <span className="text-xs text-[#666666]">Projected Winner</span>
                                        </div>
                                    </div>

                                    <span className="text-[#666666] text-sm">def.</span>

                                    {/* Loser */}
                                    <div className="flex items-center gap-2 opacity-60">
                                        {pick.loserLogo ? (
                                            <img src={pick.loserLogo} className="w-6 h-6 object-contain" />
                                        ) : (
                                            <div className="w-6 h-6 rounded-full bg-zinc-800" />
                                        )}
                                        <span className="font-medium text-[#a0a0a0]">{pick.loser}</span>
                                    </div>
                                </div>
                            </div>

                            {/* Probability Bar */}
                            <div className="col-span-5">
                                <div className="flex items-center justify-end gap-3 mb-1">
                                    <span className="text-lg font-bold text-white font-mono">{confidencePercent}</span>
                                    <span className="text-xs text-[#666666]">of 100</span>
                                </div>
                                <div className="h-2 bg-[#2a2a2a] rounded-full overflow-hidden w-full">
                                    <div
                                        className={cn("h-full rounded-full transition-all duration-500", barColor)}
                                        style={{ width: `${confidencePercent}%` }}
                                    />
                                </div>
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
