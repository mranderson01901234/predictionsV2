"use client";

import { GameDetail } from "@/lib/mock_data";


export function SpreadComparison({ game }: { game: GameDetail }) {
    const { market, prediction } = game;
    if (!market || !prediction) return null;

    const isHomeFavored = market.spread_home < 0;
    const favoredTeam = isHomeFavored ? game.home_team : game.away_team;
    const favoredLogo = isHomeFavored ? game.home_logo : game.away_logo;

    // Format spreads
    const formatSpread = (spread?: number) => {
        if (spread === undefined) return "--";
        if (spread === 0) return "PK";
        return spread > 0 ? `+${spread}` : `${spread}`;
    };

    const modelDiff = Math.abs(prediction.predicted_spread - market.spread_home);
    const highlightModel = modelDiff >= 1.5;

    return (
        <div className="bg-[#18181b]/80 backdrop-blur-sm border border-[#27272a]/50 rounded-xl p-3 shadow-lg">
            {/* Compact Vertical Layout */}
            <div className="flex flex-col gap-3">
                {/* Header + Favored Team */}
                <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-400 uppercase tracking-wider font-semibold">Pre-game</span>
                    <div className="flex items-center gap-2">
                        {favoredLogo ? (
                            <img src={favoredLogo} alt={favoredTeam} className="w-4 h-4 object-contain flex-shrink-0" />
                        ) : (
                            <div className="w-4 h-4 rounded-full bg-zinc-800 flex items-center justify-center text-[9px] text-zinc-400 flex-shrink-0">
                                {favoredTeam[0]}
                            </div>
                        )}
                        <span className="text-white font-semibold text-xs whitespace-nowrap">{favoredTeam}</span>
                        <span className="text-[9px] bg-green-500/20 text-green-400 px-1.5 py-0.5 rounded font-medium">
                            Favored
                        </span>
                    </div>
                </div>

                {/* Three Spread Boxes - Compact Horizontal */}
                <div className="grid grid-cols-3 gap-2">
                    {/* Opening */}
                    <div className="bg-[#09090b]/60 backdrop-blur-sm border border-[#27272a]/50 rounded-lg px-2 py-1.5 text-center">
                        <div className="text-[8px] text-gray-500 uppercase mb-0.5 font-medium">Opening</div>
                        <div className="text-sm text-white font-bold font-mono">{formatSpread(market.spread_home_open)}</div>
                    </div>

                    {/* Current */}
                    <div className="bg-[#09090b]/60 backdrop-blur-sm border border-[#27272a]/50 rounded-lg px-2 py-1.5 text-center">
                        <div className="text-[8px] text-gray-500 uppercase mb-0.5 font-medium">Current</div>
                        <div className="text-sm text-white font-bold font-mono">{formatSpread(market.spread_home)}</div>
                    </div>

                    {/* Model */}
                    <div className="bg-[#09090b]/60 backdrop-blur-sm border border-blue-500/40 rounded-lg px-2 py-1.5 text-center shadow-[0_0_10px_rgba(59,130,246,0.15)]">
                        <div className="text-[8px] text-gray-500 uppercase mb-0.5 font-medium">Model</div>
                        <div className="text-sm text-blue-400 font-bold font-mono">{formatSpread(Number(prediction.predicted_spread.toFixed(1)))}</div>
                    </div>
                </div>
            </div>
        </div>
    );
}
