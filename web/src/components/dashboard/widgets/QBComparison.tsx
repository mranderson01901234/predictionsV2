"use client";

import { Quarterback } from "@/lib/mock_data";
import { cn } from "@/lib/utils";

function QBCard({ qb }: { qb: Quarterback }) {
    const epaPerPlay = qb.epa / (qb.attempts || 1);
    
    return (
        <div className="flex flex-col h-full">
            {/* Top Section - Headshot + Name + Key Stats */}
            <div className="flex items-center gap-4 mb-4">
                {/* Headshot - Large Circular */}
                <div className="relative flex-shrink-0">
                    <div className="w-20 h-20 rounded-full bg-zinc-800 border-2 border-[#27272a] overflow-hidden">
                        {qb.headshot_url ? (
                            <img
                                src={qb.headshot_url}
                                alt={qb.name}
                                className="w-full h-full object-cover"
                                onError={(e) => {
                                    const target = e.target as HTMLImageElement;
                                    target.style.display = 'none';
                                    const fallback = target.nextElementSibling as HTMLElement;
                                    if (fallback) fallback.style.display = 'flex';
                                }}
                            />
                        ) : null}
                        <div className={`w-full h-full rounded-full bg-zinc-800 flex items-center justify-center text-lg text-zinc-400 ${qb.headshot_url ? 'hidden' : ''}`}>
                            {qb.name[0]}
                        </div>
                    </div>
                </div>

                {/* Name + Team + Top Stats */}
                <div className="flex-1 min-w-0">
                    <div className="text-white font-bold text-base mb-1">{qb.name}</div>
                    <div className="text-[10px] text-gray-500 font-medium mb-3">{qb.team} • QB</div>
                    
                    {/* Key Stats Row */}
                    <div className="grid grid-cols-3 gap-2">
                        <div>
                            <div className="text-[9px] text-gray-500 uppercase font-medium">TOT YDS</div>
                            <div className="text-base text-white font-bold font-mono">{qb.yards}</div>
                        </div>
                        <div>
                            <div className="text-[9px] text-gray-500 uppercase font-medium">TD</div>
                            <div className="text-base text-white font-bold font-mono">{qb.tds}</div>
                        </div>
                        <div>
                            <div className="text-[9px] text-gray-500 uppercase font-medium">INT</div>
                            <div className="text-base text-white font-bold font-mono">{qb.ints}</div>
                        </div>
                    </div>
                </div>
            </div>

            {/* EPA Section */}
            <div className="mb-4">
                <div className="flex items-center justify-between text-xs mb-2">
                    <span className="text-gray-500 font-medium">Total EPA: <span className="text-white font-bold">{qb.epa.toFixed(1)}</span></span>
                    <span className={cn("font-mono font-bold", qb.epa >= 0 ? 'text-green-400' : 'text-red-400')}>
                        {epaPerPlay.toFixed(2)} /play
                    </span>
                </div>
            </div>

            {/* Detailed Stats - Matching screenshot 2 format */}
            <div className="space-y-2 text-[10px] pt-3 border-t border-[#27272a]/50 flex-1">
                <div>
                    <div className="text-gray-500 font-medium mb-1">PASS TO</div>
                    <div className="text-white font-mono font-semibold text-xs">
                        {qb.completions}/{qb.attempts} ({qb.yards} YDS, {qb.tds} TD, {qb.ints} INT)
                    </div>
                </div>
                <div>
                    <div className="text-gray-500 font-medium mb-1">RUSH TO</div>
                    <div className="text-white font-mono font-semibold text-xs">
                        -- CAR (-- YDS, -- TD, {qb.qbr.toFixed(1)} RTG)
                    </div>
                </div>
            </div>
        </div>
    );
}

export function QBComparison({ homeQB, awayQB }: { homeQB: Quarterback, awayQB: Quarterback }) {
    return (
        <div className="bg-[#18181b]/80 backdrop-blur-sm border border-[#27272a]/50 rounded-xl p-4 shadow-lg h-full flex flex-col">
            {/* Header */}
            <div className="text-xs text-gray-400 uppercase tracking-wider mb-4 font-semibold">
                Quarterback Performance
            </div>

            {/* Two QBs side by side - Expanded */}
            <div className="grid grid-cols-2 gap-5 flex-1">
                <QBCard qb={awayQB} />
                <QBCard qb={homeQB} />
            </div>
        </div>
    );
}
