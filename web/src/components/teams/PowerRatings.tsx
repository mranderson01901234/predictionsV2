"use client";

import { cn } from "@/lib/utils";

// Mock Data for Power Ratings
const POWER_RATINGS = [
    { rank: 1, team: "BUF", logo: "https://a.espncdn.com/i/teamlogos/nfl/500/buf.png", total: 8.5, off: 6.2, def: 2.3, sos: 1.2, trend: "+0.5" },
    { rank: 2, team: "DET", logo: "https://a.espncdn.com/i/teamlogos/nfl/500/det.png", total: 7.8, off: 5.9, def: 1.9, sos: -0.5, trend: "+1.2" },
    { rank: 3, team: "KC", logo: "https://a.espncdn.com/i/teamlogos/nfl/500/kc.png", total: 7.2, off: 4.5, def: 2.7, sos: 2.1, trend: "-0.3" },
    { rank: 4, team: "PHI", logo: "https://a.espncdn.com/i/teamlogos/nfl/500/phi.png", total: 6.5, off: 4.1, def: 2.4, sos: 0.8, trend: "+0.1" },
    { rank: 5, team: "BAL", logo: "https://a.espncdn.com/i/teamlogos/nfl/500/bal.png", total: 6.1, off: 5.5, def: 0.6, sos: 1.5, trend: "-0.8" },
    { rank: 6, team: "MIN", logo: "https://a.espncdn.com/i/teamlogos/nfl/500/min.png", total: 4.8, off: 2.2, def: 2.6, sos: -0.2, trend: "+0.4" },
    { rank: 7, team: "GB", logo: "https://a.espncdn.com/i/teamlogos/nfl/500/gb.png", total: 4.2, off: 3.8, def: 0.4, sos: 0.5, trend: "+1.5" },
    { rank: 8, team: "HOU", logo: "https://a.espncdn.com/i/teamlogos/nfl/500/hou.png", total: 3.5, off: 1.5, def: 2.0, sos: -1.1, trend: "-0.2" },
    { rank: 9, team: "PIT", logo: "https://a.espncdn.com/i/teamlogos/nfl/500/pit.png", total: 3.1, off: 0.8, def: 2.3, sos: 1.8, trend: "+0.2" },
    { rank: 10, team: "WAS", logo: "https://a.espncdn.com/i/teamlogos/nfl/500/was.png", total: 2.8, off: 3.5, def: -0.7, sos: -0.8, trend: "-1.1" },
];

export function PowerRatings() {
    return (
        <div className="max-w-5xl mx-auto p-6">
            <div className="mb-8">
                <h1 className="text-2xl font-bold text-white mb-2">NFL Power Ratings</h1>
                <p className="text-[#a0a0a0]">Team strength adjusted for schedule and situation.</p>
            </div>

            <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-xl overflow-hidden">
                <div className="grid grid-cols-12 gap-4 p-4 border-b border-[#2a2a2a] text-xs text-[#666666] uppercase tracking-wider font-medium">
                    <div className="col-span-1 text-center">Rank</div>
                    <div className="col-span-4">Team</div>
                    <div className="col-span-2 text-center">Total</div>
                    <div className="col-span-1 text-center">Off</div>
                    <div className="col-span-1 text-center">Def</div>
                    <div className="col-span-1 text-center">SOS</div>
                    <div className="col-span-2 text-center">Trend</div>
                </div>

                {POWER_RATINGS.map((team) => (
                    <div key={team.team} className="grid grid-cols-12 gap-4 p-4 border-b border-[#2a2a2a] last:border-0 hover:bg-[#242424] transition-colors items-center">
                        <div className="col-span-1 text-center font-bold text-white">{team.rank}</div>

                        <div className="col-span-4 flex items-center gap-3">
                            <img src={team.logo} alt={team.team} className="w-8 h-8 object-contain" />
                            <span className="font-bold text-white">{team.team}</span>
                        </div>

                        <div className="col-span-2 text-center">
                            <span className="text-lg font-bold text-white font-mono">{team.total.toFixed(1)}</span>
                        </div>

                        <div className="col-span-1 text-center">
                            <span className={cn("font-mono font-medium", team.off > 0 ? "text-[#22c55e]" : "text-[#ef4444]")}>
                                {team.off > 0 ? "+" : ""}{team.off.toFixed(1)}
                            </span>
                        </div>

                        <div className="col-span-1 text-center">
                            <span className={cn("font-mono font-medium", team.def > 0 ? "text-[#22c55e]" : "text-[#ef4444]")}>
                                {team.def > 0 ? "+" : ""}{team.def.toFixed(1)}
                            </span>
                        </div>

                        <div className="col-span-1 text-center">
                            <span className="text-[#a0a0a0] font-mono">{team.sos > 0 ? "+" : ""}{team.sos.toFixed(1)}</span>
                        </div>

                        <div className="col-span-2 text-center">
                            <span className={cn("text-xs px-2 py-1 rounded font-medium",
                                parseFloat(team.trend) > 0 ? "bg-[#22c55e]/10 text-[#22c55e]" : "bg-[#ef4444]/10 text-[#ef4444]"
                            )}>
                                {team.trend}
                            </span>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}
