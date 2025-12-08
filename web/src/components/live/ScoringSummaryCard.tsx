"use client";

import { GameDetail } from "@/lib/mock_data";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";

interface ScoringSummaryCardProps {
    game: GameDetail;
}

export function ScoringSummaryCard({ game }: ScoringSummaryCardProps) {
    const quarters = ["1", "2", "3", "4"];

    // Determine winning team
    const awayWinning = game.away_score > game.home_score;
    const homeWinning = game.home_score > game.away_score;
    const isLive = game.status === "Live";

    return (
        <motion.div
            className="card-tertiary h-full p-4"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
        >
            {/* Compact Header */}
            <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-medium text-white/50 uppercase tracking-wider">
                    Scoring by Quarter
                </span>
                {isLive && (
                    <div className="flex items-center gap-1.5">
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                        <span className="text-[10px] text-emerald-400 font-medium">
                            Q{game.quarter}
                        </span>
                    </div>
                )}
            </div>

            {/* Score Table */}
            <table className="w-full">
                <thead>
                    <tr className="text-[9px] text-white/30 uppercase tracking-wide">
                        <th className="text-left py-1.5 font-medium">Team</th>
                        {quarters.map((q) => (
                            <th key={q} className="text-center w-8 font-medium">{q}</th>
                        ))}
                        <th className="text-right w-10 font-medium">T</th>
                    </tr>
                </thead>
                <tbody>
                    {/* Away Team Row */}
                    <tr className="border-y border-white/5">
                        <td className="py-2">
                            <div className="flex items-center gap-2">
                                {game.away_logo ? (
                                    <img
                                        src={game.away_logo}
                                        alt={game.away_team}
                                        className={cn(
                                            "w-4 h-4 object-contain",
                                            awayWinning ? "opacity-100" : "opacity-50"
                                        )}
                                    />
                                ) : (
                                    <div className={cn(
                                        "w-4 h-4 rounded-full flex items-center justify-center text-[8px] font-bold",
                                        awayWinning ? "bg-white/10 text-white/90" : "bg-white/5 text-white/40"
                                    )}>
                                        {game.away_team[0]}
                                    </div>
                                )}
                                <span className={cn(
                                    "font-medium text-xs",
                                    awayWinning ? "text-white/90" : "text-white/40"
                                )}>
                                    {game.away_team}
                                </span>
                            </div>
                        </td>
                        {game.scoring_summary.away.map((score, i) => (
                            <td key={i} className="text-center text-white/40 font-mono text-xs py-2 tabular-nums">
                                {score}
                            </td>
                        ))}
                        <td className={cn(
                            "text-right font-bold font-mono text-sm py-2 tabular-nums",
                            awayWinning ? "text-white/90" : "text-white/40"
                        )}>
                            {game.away_score}
                        </td>
                    </tr>

                    {/* Home Team Row */}
                    <tr>
                        <td className="py-2">
                            <div className="flex items-center gap-2">
                                {game.home_logo ? (
                                    <img
                                        src={game.home_logo}
                                        alt={game.home_team}
                                        className={cn(
                                            "w-4 h-4 object-contain",
                                            homeWinning ? "opacity-100" : "opacity-50"
                                        )}
                                    />
                                ) : (
                                    <div className={cn(
                                        "w-4 h-4 rounded-full flex items-center justify-center text-[8px] font-bold",
                                        homeWinning ? "bg-white/10 text-white/90" : "bg-white/5 text-white/40"
                                    )}>
                                        {game.home_team[0]}
                                    </div>
                                )}
                                <span className={cn(
                                    "font-medium text-xs",
                                    homeWinning ? "text-white/90" : "text-white/40"
                                )}>
                                    {game.home_team}
                                </span>
                            </div>
                        </td>
                        {game.scoring_summary.home.map((score, i) => (
                            <td key={i} className="text-center text-white/40 font-mono text-xs py-2 tabular-nums">
                                {score}
                            </td>
                        ))}
                        <td className={cn(
                            "text-right font-bold font-mono text-sm py-2 tabular-nums",
                            homeWinning ? "text-white/90" : "text-white/40"
                        )}>
                            {game.home_score}
                        </td>
                    </tr>
                </tbody>
            </table>

            {/* Live indicator with possession */}
            {isLive && game.time_remaining && (
                <div className="mt-3 pt-3 border-t border-white/5 flex items-center justify-center gap-2 text-[10px] text-white/40">
                    <span className="font-mono text-white/60">{game.time_remaining}</span>
                    {game.possession && (
                        <>
                            <span>•</span>
                            <span className="text-white/60">
                                {game.possession === 'home' ? game.home_team : game.away_team} ball
                            </span>
                        </>
                    )}
                </div>
            )}
        </motion.div>
    );
}

export default ScoringSummaryCard;
