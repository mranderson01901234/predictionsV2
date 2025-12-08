"use client";

import { GameDetail } from "@/lib/mock_data";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
import { TeamLogoHeader } from "./DashboardCard";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { BarChart3, ArrowDownUp, AlertOctagon } from "lucide-react";

interface TeamStatsGridProps {
    game: GameDetail;
}

interface StatRowProps {
    label: string;
    awayValue: string | number | undefined;
    homeValue: string | number | undefined;
    format?: "number" | "epa" | "percent";
    isSectionHeader?: boolean;
    highlight?: boolean;
}

function StatRow({ label, awayValue, homeValue, format = "number", isSectionHeader = false, highlight = false }: StatRowProps) {
    const formatValue = (val: string | number | undefined) => {
        if (val === undefined || val === null || val === '--') return "--";
        if (format === "epa") {
            const num = Number(val);
            if (isNaN(num)) return "--";
            return (num >= 0 ? "+" : "") + num.toFixed(2);
        }
        if (format === "percent") {
            return typeof val === "string" && val.includes("%") ? val : `${val}%`;
        }
        return val;
    };

    const getColor = (val: string | number | undefined) => {
        if (format !== "epa" || val === undefined || val === '--') return "text-[var(--foreground)]";
        const num = Number(val);
        if (isNaN(num) || num === 0) return "text-[var(--foreground)]";
        return num > 0 ? "text-[var(--neon-green)]" : "text-[var(--destructive)]";
    };

    return (
        <div className={cn(
            "flex justify-between items-center py-2 px-2 -mx-2 rounded-lg transition-colors",
            isSectionHeader && "border-t border-[var(--glass-border)] pt-3 mt-2",
            highlight && "bg-[var(--glass-bg-subtle)]",
            !isSectionHeader && "hover:bg-[var(--glass-bg-subtle)]"
        )}>
            <span className={cn(
                "text-[11px]",
                isSectionHeader 
                    ? "text-[var(--foreground)] font-semibold uppercase tracking-wide" 
                    : "text-[var(--muted-foreground)] font-medium"
            )}>
                {label}
            </span>
            <div className="flex gap-6">
                <span className={cn(
                    "text-[11px] font-mono font-semibold w-14 text-right tabular-nums",
                    getColor(awayValue)
                )}>
                    {formatValue(awayValue)}
                </span>
                <span className={cn(
                    "text-[11px] font-mono font-semibold w-14 text-right tabular-nums",
                    getColor(homeValue)
                )}>
                    {formatValue(homeValue)}
                </span>
            </div>
        </div>
    );
}

interface StatsCardProps {
    title: string;
    icon: React.ReactNode;
    game: GameDetail;
    children: React.ReactNode;
    delay?: number;
}

function StatsCard({ title, icon, game, children, delay = 0 }: StatsCardProps) {
    return (
        <motion.div 
            className="glass-surface h-full p-4"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay, duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
        >
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                    <div className="p-1.5 rounded-lg bg-[var(--glass-bg-elevated)] border border-[var(--glass-border)]">
                        {icon}
                    </div>
                    <span className="text-xs font-semibold text-[var(--foreground)] uppercase tracking-wide">
                        {title}
                    </span>
                </div>
                <TeamLogoHeader 
                    awayLogo={game.away_logo}
                    homeLogo={game.home_logo}
                    awayTeam={game.away_team}
                    homeTeam={game.home_team}
                />
            </div>
            <div className="space-y-0">
                {children}
            </div>
        </motion.div>
    );
}

// General Stats Panel
function GeneralStats({ game, delay = 0 }: { game: GameDetail; delay?: number }) {
    const parseCompAtt = (val: string | undefined) => {
        if (!val) return "--";
        return val;
    };

    return (
        <StatsCard 
            title="General" 
            icon={<BarChart3 size={12} className="text-[var(--neon-blue)]" />}
            game={game}
            delay={delay}
        >
            <StatRow label="Total Yards" awayValue={game.away_stats.total_yards} homeValue={game.home_stats.total_yards} highlight />
            <StatRow label="Total Plays" awayValue={game.away_stats.plays} homeValue={game.home_stats.plays} />
            <StatRow label="Yards / Play" awayValue={game.away_stats.yards_per_play} homeValue={game.home_stats.yards_per_play} />
            <StatRow label="EPA / Play" awayValue={game.away_stats.epa_per_play} homeValue={game.home_stats.epa_per_play} format="epa" highlight />
            
            <StatRow label="Passing" awayValue={game.away_stats.passing_yards} homeValue={game.home_stats.passing_yards} isSectionHeader />
            <StatRow label="Comp/Att" awayValue={parseCompAtt(game.away_stats.comp_att)} homeValue={parseCompAtt(game.home_stats.comp_att)} />
            <StatRow label="Yards / Pass" awayValue={game.away_stats.yards_per_pass} homeValue={game.home_stats.yards_per_pass} />
            <StatRow label="EPA / Pass" awayValue={game.away_stats.epa_per_pass} homeValue={game.home_stats.epa_per_pass} format="epa" />
            
            <StatRow label="Rushing" awayValue={game.away_stats.rushing_yards} homeValue={game.home_stats.rushing_yards} isSectionHeader />
            <StatRow label="Attempts" awayValue={game.away_stats.rush_attempts} homeValue={game.home_stats.rush_attempts} />
            <StatRow label="Yards / Rush" awayValue={game.away_stats.yards_per_rush} homeValue={game.home_stats.yards_per_rush} />
            <StatRow label="EPA / Rush" awayValue={game.away_stats.epa_per_rush} homeValue={game.home_stats.epa_per_rush} format="epa" />
        </StatsCard>
    );
}

// Downs & Conversions Panel
function DownsStats({ game, delay = 0 }: { game: GameDetail; delay?: number }) {
    const parseConversion = (val: string | undefined) => {
        if (!val) return { conv: "--", att: "--", pct: "--" };
        const match = val.match(/(\d+)\/(\d+)/);
        if (match) {
            const conv = parseInt(match[1]);
            const att = parseInt(match[2]);
            const pct = att > 0 ? `${Math.round((conv / att) * 100)}%` : "--";
            return { conv, att, pct };
        }
        return { conv: val, att: "--", pct: "--" };
    };

    const away3rd = parseConversion(game.away_stats.third_down_conv);
    const home3rd = parseConversion(game.home_stats.third_down_conv);
    const awayRZ = parseConversion(game.away_stats.red_zone_conv);
    const homeRZ = parseConversion(game.home_stats.red_zone_conv);

    return (
        <StatsCard 
            title="Conversions" 
            icon={<ArrowDownUp size={12} className="text-[var(--neon-green)]" />}
            game={game}
            delay={delay}
        >
            <StatRow label="1st Downs" awayValue={game.away_stats.first_downs} homeValue={game.home_stats.first_downs} highlight />
            <StatRow label="Passing 1D" awayValue={game.away_stats.passing_first_downs} homeValue={game.home_stats.passing_first_downs} />
            <StatRow label="Rushing 1D" awayValue={game.away_stats.rushing_first_downs} homeValue={game.home_stats.rushing_first_downs} />
            <StatRow label="Penalty 1D" awayValue={game.away_stats.penalty_first_downs || "--"} homeValue={game.home_stats.penalty_first_downs || "--"} />
            
            <StatRow label="3rd Down" awayValue={away3rd.pct} homeValue={home3rd.pct} format="percent" isSectionHeader />
            <StatRow label="Converted" awayValue={away3rd.conv} homeValue={home3rd.conv} />
            <StatRow label="Attempts" awayValue={away3rd.att} homeValue={home3rd.att} />
            
            <StatRow label="Red Zone" awayValue={awayRZ.pct} homeValue={homeRZ.pct} format="percent" isSectionHeader />
            <StatRow label="Converted" awayValue={awayRZ.conv} homeValue={homeRZ.conv} />
            <StatRow label="Attempts" awayValue={awayRZ.att} homeValue={homeRZ.att} />
        </StatsCard>
    );
}

// Negative Plays Panel
function NegativePlaysStats({ game, delay = 0 }: { game: GameDetail; delay?: number }) {
    return (
        <StatsCard 
            title="Negative Plays" 
            icon={<AlertOctagon size={12} className="text-[var(--destructive)]" />}
            game={game}
            delay={delay}
        >
            <StatRow label="Turnovers" awayValue={game.away_stats.turnovers || "--"} homeValue={game.home_stats.turnovers || "--"} highlight />
            <StatRow label="Interceptions" awayValue="--" homeValue="--" />
            <StatRow label="Fumbles Lost" awayValue="--" homeValue="--" />
            <StatRow label="Turnover EPA" awayValue="--" homeValue="--" format="epa" />
            
            <StatRow label="Sacks Allowed" awayValue={game.away_stats.sacks} homeValue={game.home_stats.sacks} isSectionHeader />
            <StatRow label="Sack Yards" awayValue={game.away_stats.sack_yards} homeValue={game.home_stats.sack_yards} />
            <StatRow label="Sack EPA" awayValue="--" homeValue="--" format="epa" />
            
            <StatRow label="Penalties" awayValue={game.away_stats.penalties || "--"} homeValue={game.home_stats.penalties || "--"} isSectionHeader />
            <StatRow label="Penalty Yards" awayValue={game.away_stats.penalty_yards || "--"} homeValue={game.home_stats.penalty_yards || "--"} />
            <StatRow label="Penalty EPA" awayValue="--" homeValue="--" format="epa" />
        </StatsCard>
    );
}

export function TeamStatsGrid({ game }: TeamStatsGridProps) {
    return (
        <div className="space-y-4">
            {/* Section Header */}
            <div className="flex items-center gap-2">
                <div className="p-2 rounded-lg bg-[var(--glass-bg-elevated)] border border-[var(--glass-border)]">
                    <BarChart3 size={14} className="text-[var(--muted-foreground)]" />
                </div>
                <div>
                    <h3 className="text-sm font-semibold text-[var(--foreground)]">
                        Team Statistics
                    </h3>
                    <p className="text-[10px] text-[var(--muted-foreground)]">
                        Detailed game breakdown
                    </p>
                </div>
            </div>
            
            {/* Mobile: Tabbed Interface */}
            <div className="lg:hidden">
                <Tabs defaultValue="general" className="w-full">
                    <TabsList className="w-full grid grid-cols-3 h-10 glass-inner p-1">
                        <TabsTrigger 
                            value="general" 
                            className="text-xs data-[state=active]:bg-[var(--glass-bg-elevated)] data-[state=active]:text-[var(--foreground)]"
                        >
                            General
                        </TabsTrigger>
                        <TabsTrigger 
                            value="downs" 
                            className="text-xs data-[state=active]:bg-[var(--glass-bg-elevated)] data-[state=active]:text-[var(--foreground)]"
                        >
                            Conversions
                        </TabsTrigger>
                        <TabsTrigger 
                            value="negative" 
                            className="text-xs data-[state=active]:bg-[var(--glass-bg-elevated)] data-[state=active]:text-[var(--foreground)]"
                        >
                            Negative
                        </TabsTrigger>
                    </TabsList>
                    <TabsContent value="general" className="mt-4">
                        <GeneralStats game={game} />
                    </TabsContent>
                    <TabsContent value="downs" className="mt-4">
                        <DownsStats game={game} />
                    </TabsContent>
                    <TabsContent value="negative" className="mt-4">
                        <NegativePlaysStats game={game} />
                    </TabsContent>
                </Tabs>
            </div>
            
            {/* Desktop: Three Card Grid */}
            <div className="hidden lg:grid lg:grid-cols-3 gap-5">
                <GeneralStats game={game} delay={0.3} />
                <DownsStats game={game} delay={0.35} />
                <NegativePlaysStats game={game} delay={0.4} />
            </div>
        </div>
    );
}

export default TeamStatsGrid;
