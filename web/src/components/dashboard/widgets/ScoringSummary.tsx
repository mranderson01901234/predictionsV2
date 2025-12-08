"use client";

import { GameDetail } from "@/lib/mock_data";
import { cn } from "@/lib/utils";

export function ScoringSummary({ game }: { game: GameDetail }) {
    const quarters = ["1", "2", "3", "4"];

    return (
        <div className="bg-[#18181b]/80 backdrop-blur-sm border border-[#27272a]/50 rounded-xl p-4 flex flex-col shadow-lg">
            {/* Header */}
            <div className="flex justify-between items-baseline mb-3">
                <div className="text-xs text-gray-400 uppercase tracking-wider font-semibold">
                    Scoring Summary
                </div>
                <div className="text-[10px] text-gray-500 font-medium">Box Score</div>
            </div>

            {/* Compact Table */}
            <table className="w-full">
                <thead>
                    <tr className="text-[10px] text-gray-500 border-b border-[#27272a]/50">
                        <th className="text-left py-1.5 font-medium">Team</th>
                        <th className="text-center w-7 font-medium">1</th>
                        <th className="text-center w-7 font-medium">2</th>
                        <th className="text-center w-7 font-medium">3</th>
                        <th className="text-center w-7 font-medium">4</th>
                        <th className="text-right w-10 font-medium">T</th>
                    </tr>
                </thead>
                <tbody>
                    {/* Away Team Row */}
                    <tr className="border-b border-[#27272a]/30">
                        <td className="py-2 flex items-center gap-2">
                            {game.away_logo ? (
                                <img src={game.away_logo} alt={game.away_team} className="w-4 h-4 object-contain flex-shrink-0" />
                            ) : (
                                <div className="w-4 h-4 rounded-full bg-zinc-800 flex items-center justify-center text-[8px] text-zinc-400 flex-shrink-0">
                                    {game.away_team[0]}
                                </div>
                            )}
                            <span className="text-white font-semibold text-xs">{game.away_team}</span>
                        </td>
                        {game.scoring_summary.away.map((score, i) => (
                            <td key={i} className="text-center text-gray-300 font-mono text-xs py-2">{score}</td>
                        ))}
                        <td className="text-right text-white font-bold font-mono text-sm py-2">{game.away_score}</td>
                    </tr>
                    {/* Home Team Row */}
                    <tr>
                        <td className="py-2 flex items-center gap-2">
                            {game.home_logo ? (
                                <img src={game.home_logo} alt={game.home_team} className="w-4 h-4 object-contain flex-shrink-0" />
                            ) : (
                                <div className="w-4 h-4 rounded-full bg-zinc-800 flex items-center justify-center text-[8px] text-zinc-400 flex-shrink-0">
                                    {game.home_team[0]}
                                </div>
                            )}
                            <span className="text-white font-semibold text-xs">{game.home_team}</span>
                        </td>
                        {game.scoring_summary.home.map((score, i) => (
                            <td key={i} className="text-center text-gray-300 font-mono text-xs py-2">{score}</td>
                        ))}
                        <td className="text-right text-white font-bold font-mono text-sm py-2">{game.home_score}</td>
                    </tr>
                </tbody>
            </table>

            {/* Footer */}
            <div className="text-right text-[10px] text-gray-500 mt-2 font-medium">
                {game.status === 'Final' ? 'Final' : game.status === 'Live' ? 'Live' : 'Scheduled'}
            </div>
        </div>
    );
}
