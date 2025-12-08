"use client";

import { useState, useEffect } from "react";
import { Game, GameDetail } from "@/lib/mock_data";
import { motion, AnimatePresence } from "framer-motion";

// Live Dashboard Components
import { LiveLayoutShell } from "./LiveLayoutShell";
import { GameStrip } from "./GameStrip";
import { HeroGameCard } from "./HeroGameCard";
import { ScoringSummaryCard } from "./ScoringSummaryCard";
import { WinProbabilityCard } from "./WinProbabilityCard";
import { QuarterbackPerformanceCard } from "./QuarterbackPerformanceCard";
import { PreGameLineCard } from "./PreGameLineCard";
import { TeamStatsGrid } from "./TeamStatsGrid";
import { AIIntelligenceRail } from "./AIIntelligenceRail";
import { 
    DashboardCardSkeleton, 
    ChartSkeleton, 
    QBCardSkeleton 
} from "./DashboardCard";

interface LiveDashboardProps {
    games: Game[];
    initialDetails: Record<string, GameDetail>;
}

export function LiveDashboard({ games, initialDetails }: LiveDashboardProps) {
    const [selectedGameId, setSelectedGameId] = useState<string>(games[0]?.game_id);
    const [lastUpdated, setLastUpdated] = useState<string>("");
    const [isLoading, setIsLoading] = useState(false);
    
    const selectedGame = initialDetails[selectedGameId];

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
        // Simulate data fetch
        setTimeout(() => setIsLoading(false), 200);
    };

    if (!selectedGame) {
        return (
            <LiveLayoutShell lastUpdated={lastUpdated}>
                <div className="flex-1 flex items-center justify-center">
                    <p className="text-[var(--muted-foreground)]">No games available</p>
                </div>
            </LiveLayoutShell>
        );
    }

    return (
        <LiveLayoutShell lastUpdated={lastUpdated}>
            {/* Game Strip */}
            <GameStrip
                games={games}
                selectedGameId={selectedGameId}
                onSelect={handleGameSelect}
            />

            {/* Main Content Area */}
            <div className="flex-1 overflow-y-auto">
                <AnimatePresence mode="wait">
                    <motion.div
                        key={selectedGameId}
                        initial={{ opacity: 0, y: 8 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -8 }}
                        transition={{ duration: 0.2, ease: "easeOut" }}
                        className="content-container page-padding py-6"
                    >
                        {isLoading ? (
                            <LoadingSkeleton />
                        ) : (
                            <div className="flex flex-col xl:flex-row gap-6">
                                {/* Main Content Column */}
                                <div className="flex-1 min-w-0 space-y-6">
                                    {/* === ZONE 1: HERO === */}
                                    <section className="zone-hero animate-zone-reveal">
                                        <HeroGameCard game={selectedGame} />
                                    </section>

                                    {/* === ZONE 2: ANALYTICS === */}
                                    <section className="zone-analytics">
                                        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
                                            {/* Left: Scoring Summary + Pre-game Lines */}
                                            <div className="lg:col-span-3 flex flex-col gap-5">
                                                <motion.div
                                                    initial={{ opacity: 0, y: 12 }}
                                                    animate={{ opacity: 1, y: 0 }}
                                                    transition={{ delay: 0.1, duration: 0.4 }}
                                                >
                                                    <ScoringSummaryCard game={selectedGame} />
                                                </motion.div>
                                                <motion.div
                                                    initial={{ opacity: 0, y: 12 }}
                                                    animate={{ opacity: 1, y: 0 }}
                                                    transition={{ delay: 0.15, duration: 0.4 }}
                                                >
                                                    <PreGameLineCard game={selectedGame} />
                                                </motion.div>
                                            </div>

                                            {/* Center: Win Probability Chart */}
                                            <div className="lg:col-span-4">
                                                <WinProbabilityCard game={selectedGame} />
                                            </div>

                                            {/* Right: QB Performance */}
                                            <div className="lg:col-span-5">
                                                <QuarterbackPerformanceCard
                                                    homeQB={selectedGame.home_qb}
                                                    awayQB={selectedGame.away_qb}
                                                />
                                            </div>
                                        </div>
                                    </section>

                                    {/* === ZONE 4: DEEP STATS === */}
                                    <section className="zone-deep-stats">
                                        <motion.div
                                            initial={{ opacity: 0, y: 12 }}
                                            animate={{ opacity: 1, y: 0 }}
                                            transition={{ delay: 0.3, duration: 0.4 }}
                                        >
                                            <TeamStatsGrid game={selectedGame} />
                                        </motion.div>
                                    </section>
                                </div>

                                {/* === ZONE 3: AI INTELLIGENCE RAIL === */}
                                <aside className="w-full xl:w-80 xl:flex-shrink-0">
                                    <div className="xl:sticky xl:top-20">
                                        <AIIntelligenceRail game={selectedGame} />
                                    </div>
                                </aside>
                            </div>
                        )}
                    </motion.div>
                </AnimatePresence>
            </div>
        </LiveLayoutShell>
    );
}

// Loading skeleton component
function LoadingSkeleton() {
    return (
        <div className="flex flex-col xl:flex-row gap-6">
            {/* Main Column Skeleton */}
            <div className="flex-1 min-w-0 space-y-6">
                {/* Hero Skeleton */}
                <div className="glass-surface-hero p-6 h-[200px] skeleton-shimmer rounded-2xl" />
                
                {/* Analytics Skeleton */}
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
                    <div className="lg:col-span-3 space-y-5">
                        <DashboardCardSkeleton className="h-[180px]" />
                        <DashboardCardSkeleton className="h-[140px]" />
                    </div>
                    <div className="lg:col-span-4">
                        <ChartSkeleton className="h-full min-h-[320px]" />
                    </div>
                    <div className="lg:col-span-5">
                        <QBCardSkeleton className="h-full min-h-[320px]" />
                    </div>
                </div>
                
                {/* Stats Skeleton */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
                    <DashboardCardSkeleton className="h-[300px]" />
                    <DashboardCardSkeleton className="h-[300px]" />
                    <DashboardCardSkeleton className="h-[300px]" />
                </div>
            </div>
            
            {/* AI Rail Skeleton */}
            <div className="w-full xl:w-80 xl:flex-shrink-0">
                <div className="glass-rail p-4 h-[500px] skeleton-shimmer rounded-2xl" />
            </div>
        </div>
    );
}

export default LiveDashboard;
