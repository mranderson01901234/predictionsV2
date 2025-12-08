import { getGameDetails } from "@/lib/mock_data";
import { AnalyticsChart } from "@/components/game-detail/AnalyticsChart";
import { PredictionCard } from "@/components/game-detail/PredictionCard";
import { GlassCard } from "@/components/ui/glass";
import { ArrowLeft, Calendar, Clock, MapPin } from "lucide-react";
import Link from "next/link";
import { notFound } from "next/navigation";

interface PageProps {
    params: Promise<{ id: string }>;
}

export default async function GameDetailPage({ params }: PageProps) {
    const { id } = await params;
    const data = await getGameDetails(id);

    if (!data.game) {
        notFound();
    }

    const { game, market, prediction } = data;

    return (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            {/* Back Link */}
            <Link
                href="/games"
                className="inline-flex items-center gap-2 text-slate-400 hover:text-white mb-6 transition-colors"
            >
                <ArrowLeft className="w-4 h-4" />
                Back to Dashboard
            </Link>

            {/* Hero Section */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
                <GlassCard className="lg:col-span-2 flex flex-col justify-between relative overflow-hidden">
                    {/* Background Team Logos/Colors (Abstract) */}
                    <div className="absolute inset-0 bg-gradient-to-br from-slate-900 via-slate-900 to-slate-800 opacity-50" />

                    <div className="relative z-10">
                        <div className="flex justify-between items-start mb-8">
                            <div className="flex items-center gap-2 text-slate-400 text-sm">
                                <Calendar className="w-4 h-4" />
                                {new Date(game.date).toLocaleDateString(undefined, { weekday: 'long', month: 'short', day: 'numeric' })}
                                <span className="mx-2">•</span>
                                <Clock className="w-4 h-4" />
                                {new Date(game.date).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                            </div>
                            <div className="px-3 py-1 rounded-full bg-slate-800/50 border border-slate-700 text-xs font-mono text-slate-300">
                                Week {game.week}
                            </div>
                        </div>

                        <div className="flex items-center justify-between gap-8">
                            {/* Away Team */}
                            <div className="text-center">
                                <div className="text-4xl md:text-6xl font-black text-white mb-2">{game.away_team}</div>
                                <div className="text-slate-400 font-medium">Away</div>
                            </div>

                            <div className="text-2xl font-mono text-slate-500">@</div>

                            {/* Home Team */}
                            <div className="text-center">
                                <div className="text-4xl md:text-6xl font-black text-white mb-2">{game.home_team}</div>
                                <div className="text-slate-400 font-medium">Home</div>
                            </div>
                        </div>
                    </div>
                </GlassCard>

                {/* Prediction Card */}
                <div className="lg:col-span-1">
                    {prediction && market ? (
                        <PredictionCard prediction={prediction} marketSpread={market.spread_home} />
                    ) : (
                        <GlassCard className="h-full flex items-center justify-center text-slate-500">
                            No prediction available
                        </GlassCard>
                    )}
                </div>
            </div>

            {/* Analytics Section */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Main Chart */}
                <div className="lg:col-span-2">
                    {market && prediction ? (
                        <AnalyticsChart
                            marketSpread={market.spread_home}
                            modelSpread={prediction.predicted_spread}
                        />
                    ) : (
                        <GlassCard className="h-[400px] flex items-center justify-center text-slate-500">
                            Chart unavailable
                        </GlassCard>
                    )}
                </div>

                {/* Stats Grid */}
                <div className="space-y-4">
                    <GlassCard>
                        <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4">Team Stats (EPA/Play)</h3>
                        <div className="space-y-4">
                            <div className="flex justify-between items-center">
                                <span className="font-bold text-white">{game.away_team}</span>
                                <div className="flex gap-4 text-sm font-mono">
                                    <span className="text-emerald-400">Off: +0.12</span>
                                    <span className="text-rose-400">Def: +0.05</span>
                                </div>
                            </div>
                            <div className="w-full h-px bg-slate-800" />
                            <div className="flex justify-between items-center">
                                <span className="font-bold text-white">{game.home_team}</span>
                                <div className="flex gap-4 text-sm font-mono">
                                    <span className="text-emerald-400">Off: +0.08</span>
                                    <span className="text-emerald-400">Def: -0.03</span>
                                </div>
                            </div>
                        </div>
                    </GlassCard>

                    <GlassCard>
                        <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4">Market Consensus</h3>
                        <div className="space-y-3">
                            <div className="flex justify-between text-sm">
                                <span className="text-slate-400">Spread</span>
                                <span className="text-white font-mono">{market?.spread_home}</span>
                            </div>
                            <div className="flex justify-between text-sm">
                                <span className="text-slate-400">Total</span>
                                <span className="text-white font-mono">{market?.total}</span>
                            </div>
                            <div className="flex justify-between text-sm">
                                <span className="text-slate-400">Public %</span>
                                <span className="text-white font-mono">62% on Home</span>
                            </div>
                        </div>
                    </GlassCard>
                </div>
            </div>
        </div>
    );
}
