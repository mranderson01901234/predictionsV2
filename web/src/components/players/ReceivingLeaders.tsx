"use client";

import { cn } from "@/lib/utils";

// Mock Data for Receiving Leaders
const RECEIVING_LEADERS = [
    { rank: 1, name: "Justin Jefferson", team: "MIN", logo: "https://a.espncdn.com/i/teamlogos/nfl/500/min.png", rec: 88, yds: 1250, tds: 8, tgt: 115, share: "28%" },
    { rank: 2, name: "Tyreek Hill", team: "MIA", logo: "https://a.espncdn.com/i/teamlogos/nfl/500/mia.png", rec: 82, yds: 1180, tds: 10, tgt: 108, share: "31%" },
    { rank: 3, name: "Ja'Marr Chase", team: "CIN", logo: "https://a.espncdn.com/i/teamlogos/nfl/500/cin.png", rec: 78, yds: 1120, tds: 9, tgt: 105, share: "29%" },
    { rank: 4, name: "CeeDee Lamb", team: "DAL", logo: "https://a.espncdn.com/i/teamlogos/nfl/500/dal.png", rec: 85, yds: 1090, tds: 6, tgt: 118, share: "27%" },
    { rank: 5, name: "A.J. Brown", team: "PHI", logo: "https://a.espncdn.com/i/teamlogos/nfl/500/phi.png", rec: 72, yds: 1050, tds: 7, tgt: 98, share: "26%" },
    { rank: 6, name: "Amon-Ra St. Brown", team: "DET", logo: "https://a.espncdn.com/i/teamlogos/nfl/500/det.png", rec: 80, yds: 980, tds: 5, tgt: 102, share: "25%" },
    { rank: 7, name: "Puka Nacua", team: "LAR", logo: "https://a.espncdn.com/i/teamlogos/nfl/500/lar.png", rec: 75, yds: 950, tds: 4, tgt: 95, share: "24%" },
    { rank: 8, name: "Garrett Wilson", team: "NYJ", logo: "https://a.espncdn.com/i/teamlogos/nfl/500/nyj.png", rec: 68, yds: 920, tds: 3, tgt: 92, share: "23%" },
    { rank: 9, name: "Stefon Diggs", team: "HOU", logo: "https://a.espncdn.com/i/teamlogos/nfl/500/hou.png", rec: 70, yds: 890, tds: 6, tgt: 90, share: "22%" },
    { rank: 10, name: "Davante Adams", team: "NYJ", logo: "https://a.espncdn.com/i/teamlogos/nfl/500/nyj.png", rec: 65, yds: 850, tds: 5, tgt: 88, share: "21%" },
];

export function ReceivingLeaders() {
    return (
        <div className="max-w-5xl mx-auto p-6">
            <div className="mb-8">
                <h1 className="text-2xl font-bold text-white mb-2">Receiving Leaders</h1>
                <p className="text-[#a0a0a0]">Yards, Targets, and Target Share</p>
            </div>

            <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-xl overflow-hidden">
                <div className="grid grid-cols-12 gap-4 p-4 border-b border-[#2a2a2a] text-xs text-[#666666] uppercase tracking-wider font-medium">
                    <div className="col-span-1 text-center">Rank</div>
                    <div className="col-span-5">Player</div>
                    <div className="col-span-1 text-center">Rec</div>
                    <div className="col-span-2 text-center">Yards</div>
                    <div className="col-span-1 text-center">TD</div>
                    <div className="col-span-1 text-center">Tgt</div>
                    <div className="col-span-1 text-center">Share</div>
                </div>

                {RECEIVING_LEADERS.map((player) => (
                    <div key={player.name} className="grid grid-cols-12 gap-4 p-4 border-b border-[#2a2a2a] last:border-0 hover:bg-[#242424] transition-colors items-center">
                        <div className="col-span-1 text-center font-bold text-white">{player.rank}</div>

                        <div className="col-span-5 flex items-center gap-3">
                            <div className="w-10 h-10 rounded-full bg-[#242424] overflow-hidden border border-[#2a2a2a]">
                                <img src={`https://a.espncdn.com/combiner/i?img=/i/headshots/nfl/players/full/4241464.png&w=350&h=254`} alt={player.name} className="w-full h-full object-cover" />
                            </div>
                            <div className="flex flex-col">
                                <span className="font-bold text-white">{player.name}</span>
                                <div className="flex items-center gap-1 text-xs text-[#666666]">
                                    <img src={player.logo} className="w-3 h-3 object-contain" />
                                    <span>{player.team}</span>
                                </div>
                            </div>
                        </div>

                        <div className="col-span-1 text-center">
                            <span className="text-white font-mono">{player.rec}</span>
                        </div>

                        <div className="col-span-2 text-center">
                            <span className="text-lg font-bold text-white font-mono">{player.yds}</span>
                        </div>

                        <div className="col-span-1 text-center">
                            <span className="text-white font-mono">{player.tds}</span>
                        </div>

                        <div className="col-span-1 text-center">
                            <span className="text-[#a0a0a0] font-mono">{player.tgt}</span>
                        </div>

                        <div className="col-span-1 text-center">
                            <span className="text-[#3b82f6] font-bold font-mono">{player.share}</span>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}
