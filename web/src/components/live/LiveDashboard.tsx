"use client";

import { useState, useEffect } from "react";
import { Game, GameDetail } from "@/lib/mock_data";
import { motion, AnimatePresence } from "framer-motion";
import { ChatContext } from "@/lib/ai/context-builder";
import { getPrediction, Prediction } from "@/lib/api/predictions";

// Live Dashboard Components
import { LiveLayoutShell } from "./LiveLayoutShell";
import { GameStrip } from "./GameStrip";
import { DashboardLayout } from "@/components/layout/DashboardLayout";

// New Main Content Components
import { HeroScore } from "./HeroScore";
import { KeyStatsStrip } from "./KeyStatsStrip";
import { QBComparison } from "./QBComparison";
import { TeamStatistics } from "./TeamStatistics";

interface LiveDashboardProps {
    games: Game[];
    initialDetails: Record<string, GameDetail>;
}

export function LiveDashboard({ games, initialDetails }: LiveDashboardProps) {
    const [selectedGameId, setSelectedGameId] = useState<string>(games[0]?.game_id);
    const [lastUpdated, setLastUpdated] = useState<string>("");
    const [isLoading, setIsLoading] = useState(false);
    const [apiPrediction, setApiPrediction] = useState<Prediction | null>(null);

    const selectedGame = initialDetails[selectedGameId];

    // Fetch real prediction when game changes
    useEffect(() => {
        if (!selectedGame) return;
        const fetchPrediction = async () => {
            try {
                const pred = await getPrediction(selectedGame.game_id);
                setApiPrediction(pred);
            } catch (error) {
                // Only log unexpected errors (404s return null, not throw)
                console.warn("Failed to fetch prediction:", error);
                setApiPrediction(null);
            }
        };
        fetchPrediction();
    }, [selectedGame?.game_id]);

    // Build chat context from game details
    const chatContext: ChatContext | undefined = selectedGame ? {
        game: selectedGame,
        prediction: apiPrediction || selectedGame.prediction,
        userBet: undefined,
    } : undefined;

    // Format last updated time
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
        const interval = setInterval(updateTime, 60000);
        return () => clearInterval(interval);
    }, []);

    // Simulate loading state on game change
    const handleGameSelect = (gameId: string) => {
        setIsLoading(true);
        setSelectedGameId(gameId);
        setTimeout(() => setIsLoading(false), 200);
    };

    if (!selectedGame) {
        return (
            <LiveLayoutShell lastUpdated={lastUpdated}>
                <div className="flex-1 flex items-center justify-center">
                    <p className="text-white/40">No games available</p>
                </div>
            </LiveLayoutShell>
        );
    }

    return (
        <LiveLayoutShell lastUpdated={lastUpdated}>
            <DashboardLayout gameContext={chatContext}>
                <div className="flex h-full w-full">
                    {/* Left: Vertical Game Strip */}
                    <GameStrip
                        games={games}
                        selectedGameId={selectedGameId}
                        onSelect={handleGameSelect}
                    />

                    {/* Main Content Area - Fits viewport, no scrolling */}
                    <div className="flex-1 overflow-hidden min-w-0 flex flex-col">
                        <AnimatePresence mode="wait">
                            <motion.div
                                key={selectedGameId}
                                initial={{ opacity: 0, y: 8 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: -8 }}
                                transition={{ duration: 0.2, ease: "easeOut" }}
                                className="flex-1 overflow-hidden min-h-0 flex flex-col"
                            >
                                <div className="flex-1 overflow-hidden flex flex-col px-4 py-2 max-w-[1200px] mx-auto w-full">
                                    {isLoading ? (
                                        <LoadingSkeleton />
                                    ) : (
                                        <div className="flex-1 overflow-hidden flex flex-col gap-2">
                                        {/* Hero Score */}
                                        <div className="flex-shrink-0">
                                            <HeroScore game={selectedGame} />
                                        </div>

                                        {/* Key Stats Strip */}
                                        <div className="flex-shrink-0">
                                            <KeyStatsStrip game={selectedGame} />
                                        </div>

                                        {/* QB + Detailed Stats - Takes remaining space */}
                                        <div className="flex-1 min-h-0 grid grid-cols-1 lg:grid-cols-12 gap-3">
                                            {/* QB Comparison - Takes 5 columns for premium cards */}
                                            <div className="lg:col-span-5 overflow-hidden">
                                                <QBComparison game={selectedGame} />
                                            </div>

                                            {/* Team Statistics - Takes 7 columns */}
                                            <div className="lg:col-span-7 overflow-hidden">
                                                <TeamStatistics game={selectedGame} />
                                            </div>
                                        </div>
                                        </div>
                                    )}
                                </div>
                            </motion.div>
                        </AnimatePresence>
                    </div>
                </div>
            </DashboardLayout>
        </LiveLayoutShell>
    );
}

// Loading skeleton component
function LoadingSkeleton() {
    return (
        <div className="space-y-4">
            {/* Hero Score Skeleton */}
            <div className="h-[140px] skeleton-shimmer rounded-2xl" />

            {/* Key Stats Strip Skeleton */}
            <div className="h-[70px] skeleton-shimmer rounded-xl" />

            {/* QB + Stats Grid Skeleton */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
                {/* QB Comparison Skeleton */}
                <div className="lg:col-span-5 h-[300px] skeleton-shimmer rounded-2xl" />

                {/* Team Statistics Skeleton */}
                <div className="lg:col-span-7 h-[300px] skeleton-shimmer rounded-2xl" />
            </div>
        </div>
    );
}

export default LiveDashboard;
