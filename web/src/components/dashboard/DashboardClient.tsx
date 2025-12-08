"use client";

import { useState, useEffect } from "react";
import { Game, GameDetail } from "@/lib/mock_data";
import { LiveScoreboard } from "@/components/dashboard/LiveScoreboard";
import { ScoringSummary } from "@/components/dashboard/widgets/ScoringSummary";
import { QBComparison } from "@/components/dashboard/widgets/QBComparison";
import { SpreadComparison } from "@/components/dashboard/widgets/SpreadComparison";
import { WinProbabilityChart } from "@/components/game-detail/WinProbabilityChart";
import { TeamStatsTable } from "@/components/dashboard/widgets/TeamStats";
import { motion, AnimatePresence } from "framer-motion";

interface DashboardClientProps {
    games: Game[];
    initialDetails: Record<string, GameDetail>;
}

export function DashboardClient({ games, initialDetails }: DashboardClientProps) {
    const [selectedGameId, setSelectedGameId] = useState<string>(games[0]?.game_id);
    const [lastUpdated, setLastUpdated] = useState<string>("");
    const selectedGame = initialDetails[selectedGameId];

    useEffect(() => {
        const updateTime = () => {
            const now = new Date();
            const formatted = now.toLocaleString('en-US', { 
                month: 'short', 
                day: 'numeric', 
                hour: 'numeric', 
                minute: '2-digit',
                hour12: true 
            });
            setLastUpdated(formatted);
        };
        updateTime();
        const interval = setInterval(updateTime, 60000); // Update every minute
        return () => clearInterval(interval);
    }, []);

    if (!selectedGame) return <div>Loading...</div>;

    return (
        <div className="h-[calc(100vh-4rem)] flex flex-col max-w-[1920px] mx-auto px-0 py-0 overflow-hidden bg-[#0a0a0a]">
            {/* Header Section */}
            <section className="flex-none w-full border-b border-[#2a2a2a] bg-[#0a0a0a] px-6 py-4">
                <div className="flex items-center justify-between">
                    <div>
                        <div className="flex items-center gap-2 text-sm text-gray-400 mb-1">
                            <span>Predictr</span>
                            <span>/</span>
                            <span className="text-white">live</span>
                        </div>
                        <h1 className="text-2xl font-bold text-white">Live Scoreboard</h1>
                        {lastUpdated && (
                            <p className="text-xs text-gray-500 mt-1">Last updated: {lastUpdated}</p>
                        )}
                    </div>
                </div>
            </section>

            {/* Top Bar: Live Scoreboard - Fixed Height */}
            <section className="flex-none w-full">
                <LiveScoreboard
                    games={games}
                    selectedGameId={selectedGameId}
                    onSelect={setSelectedGameId}
                />
            </section>

            {/* Main Content - Scrollable */}
            <AnimatePresence mode="wait">
                <motion.div
                    key={selectedGameId}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    transition={{ duration: 0.2 }}
                    className="flex-1 min-h-0 overflow-y-auto pb-20 lg:pb-6 p-6"
                >
                    {/* ROW 1: Scoring Summary + Pre-game, Win Probability, QB Performance */}
                    <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 mb-6">
                        {/* Left Column: Scoring Summary + Pre-game stacked */}
                        <div className="lg:col-span-2 flex flex-col gap-5">
                            <ScoringSummary game={selectedGame} />
                            <SpreadComparison game={selectedGame} />
                        </div>

                        {/* Middle Column: Win Probability */}
                        <div className="lg:col-span-5">
                            <WinProbabilityChart game={selectedGame} />
                        </div>

                        {/* Right Column: QB Performance - Taller */}
                        <div className="lg:col-span-5">
                            <QBComparison
                                homeQB={selectedGame.home_qb}
                                awayQB={selectedGame.away_qb}
                            />
                        </div>
                    </div>

                    {/* ROW 3: Team Statistics Table */}
                    <TeamStatsTable game={selectedGame} />
                </motion.div>
            </AnimatePresence>
        </div>
    );
}
