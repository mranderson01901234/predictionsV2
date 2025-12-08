"use client";

import { cn } from "@/lib/utils";

// Mock Data for QB Rankings
const QB_RANKINGS = [
    { rank: 1, name: "Josh Allen", team: "BUF", logo: "https://a.espncdn.com/i/teamlogos/nfl/500/buf.png", epa: 0.35, cpoe: 5.2, qbr: 78.4, plays: 450 },
    { rank: 2, name: "Lamar Jackson", team: "BAL", logo: "https://a.espncdn.com/i/teamlogos/nfl/500/bal.png", epa: 0.31, cpoe: 4.8, qbr: 72.1, plays: 420 },
    { rank: 3, name: "Patrick Mahomes", team: "KC", logo: "https://a.espncdn.com/i/teamlogos/nfl/500/kc.png", epa: 0.28, cpoe: 3.5, qbr: 70.5, plays: 480 },
    { rank: 4, name: "Jalen Hurts", team: "PHI", logo: "https://a.espncdn.com/i/teamlogos/nfl/500/phi.png", epa: 0.25, cpoe: 2.1, qbr: 68.9, plays: 430 },
    { rank: 5, name: "Jared Goff", team: "DET", logo: "https://a.espncdn.com/i/teamlogos/nfl/500/det.png", epa: 0.24, cpoe: 6.5, qbr: 67.2, plays: 410 },
    { rank: 6, name: "Joe Burrow", team: "CIN", logo: "https://a.espncdn.com/i/teamlogos/nfl/500/cin.png", epa: 0.22, cpoe: 4.1, qbr: 65.8, plays: 460 },
    { rank: 7, name: "C.J. Stroud", team: "HOU", logo: "https://a.espncdn.com/i/teamlogos/nfl/500/hou.png", epa: 0.18, cpoe: 1.2, qbr: 62.4, plays: 440 },
    { rank: 8, name: "Jordan Love", team: "GB", logo: "https://a.espncdn.com/i/teamlogos/nfl/500/gb.png", epa: 0.16, cpoe: 0.8, qbr: 60.1, plays: 400 },
    { rank: 9, name: "Baker Mayfield", team: "TB", logo: "https://a.espncdn.com/i/teamlogos/nfl/500/tb.png", epa: 0.14, cpoe: 1.5, qbr: 58.7, plays: 425 },
    { rank: 10, name: "Kyler Murray", team: "ARI", logo: "https://a.espncdn.com/i/teamlogos/nfl/500/ari.png", epa: 0.12, cpoe: -0.5, qbr: 56.3, plays: 390 },
];

export function QBRankings() {
    return (
        <div className="max-w-5xl mx-auto p-6">
            <div className="mb-8">
                <h1 className="text-2xl font-bold text-white mb-2">Quarterback Rankings</h1>
                <p className="text-[#a0a0a0]">EPA/Play + CPOE Composite Rankings</p>
            </div>

            <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-xl overflow-hidden">
                <div className="grid grid-cols-12 gap-4 p-4 border-b border-[#2a2a2a] text-xs text-[#666666] uppercase tracking-wider font-medium">
                    <div className="col-span-1 text-center">Rank</div>
                    <div className="col-span-5">Player</div>
                    <div className="col-span-2 text-center">EPA/Play</div>
                    <div className="col-span-2 text-center">CPOE</div>
                    <div className="col-span-2 text-center">QBR</div>
                </div>

                {QB_RANKINGS.map((qb) => (
                    <div key={qb.name} className="grid grid-cols-12 gap-4 p-4 border-b border-[#2a2a2a] last:border-0 hover:bg-[#242424] transition-colors items-center">
                        <div className="col-span-1 text-center font-bold text-white">{qb.rank}</div>

                        <div className="col-span-5 flex items-center gap-3">
                            <div className="w-10 h-10 rounded-full bg-[#242424] overflow-hidden border border-[#2a2a2a]">
                                <img src={`https://a.espncdn.com/combiner/i?img=/i/headshots/nfl/players/full/3918298.png&w=350&h=254`} alt={qb.name} className="w-full h-full object-cover" />
                            </div>
                            <div className="flex flex-col">
                                <span className="font-bold text-white">{qb.name}</span>
                                <div className="flex items-center gap-1 text-xs text-[#666666]">
                                    <img src={qb.logo} className="w-3 h-3 object-contain" />
                                    <span>{qb.team}</span>
                                </div>
                            </div>
                        </div>

                        <div className="col-span-2 text-center">
                            <span className={cn("text-lg font-bold font-mono", qb.epa > 0 ? "text-[#22c55e]" : "text-[#ef4444]")}>
                                {qb.epa > 0 ? "+" : ""}{qb.epa.toFixed(2)}
                            </span>
                        </div>

                        <div className="col-span-2 text-center">
                            <span className={cn("font-mono font-medium", qb.cpoe > 0 ? "text-[#22c55e]" : "text-[#ef4444]")}>
                                {qb.cpoe > 0 ? "+" : ""}{qb.cpoe.toFixed(1)}%
                            </span>
                        </div>

                        <div className="col-span-2 text-center">
                            <span className="text-white font-mono font-bold">{qb.qbr.toFixed(1)}</span>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}
