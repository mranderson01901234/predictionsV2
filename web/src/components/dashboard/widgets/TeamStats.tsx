"use client";

import { GameDetail } from "@/lib/mock_data";
import { cn } from "@/lib/utils";

function StatRow({ label, awayValue, homeValue, format = "number", isHeader = false }: {
    label: string,
    awayValue: string | number | undefined,
    homeValue: string | number | undefined,
    format?: "number" | "epa" | "percent",
    isHeader?: boolean
}) {
    const formatValue = (val: string | number | undefined) => {
        if (val === undefined || val === null || val === '--') return "--";
        if (format === "epa") {
            const num = Number(val);
            return (num >= 0 ? "+" : "") + num.toFixed(2);
        }
        return val;
    };

    const getColor = (val: string | number | undefined) => {
        if (format !== "epa" || val === undefined || val === '--') return "text-gray-300";
        const num = Number(val);
        if (num === 0) return "text-gray-300";
        return num > 0 ? "text-green-400" : "text-red-400";
    };

    return (
        <div className={cn(
            "flex justify-between py-1.5",
            isHeader && "border-t border-[#27272a]/50 pt-3 mt-2"
        )}>
            <span className={cn(
                "text-xs",
                isHeader ? "text-gray-300 font-semibold" : "text-gray-500 font-medium"
            )}>
                {label}
            </span>
            <div className="flex gap-6">
                <span className={cn("text-xs w-14 text-right font-mono font-semibold", getColor(awayValue))}>
                    {formatValue(awayValue)}
                </span>
                <span className={cn("text-xs w-14 text-right font-mono font-semibold", getColor(homeValue))}>
                    {formatValue(homeValue)}
                </span>
            </div>
        </div>
    );
}

function StatGroup({ title, children, awayLogo, homeLogo }: { title: string, children: React.ReactNode, awayLogo?: string, homeLogo?: string }) {
    return (
        <div className="bg-[#18181b]/80 backdrop-blur-sm border border-[#27272a]/50 rounded-xl p-4 flex flex-col shadow-lg max-h-[400px]">
            {/* Header with team logos */}
            <div className="flex items-center justify-between mb-3 flex-shrink-0">
                <span className="text-sm text-gray-400 font-semibold">{title}</span>
                <div className="flex gap-3">
                    {awayLogo ? (
                        <img src={awayLogo} alt="Away" className="w-4 h-4 object-contain" />
                    ) : (
                        <div className="w-4 h-4 rounded-full bg-zinc-800 flex items-center justify-center text-[9px] text-zinc-400">A</div>
                    )}
                    {homeLogo ? (
                        <img src={homeLogo} alt="Home" className="w-4 h-4 object-contain" />
                    ) : (
                        <div className="w-4 h-4 rounded-full bg-zinc-800 flex items-center justify-center text-[9px] text-zinc-400">H</div>
                    )}
                </div>
            </div>
            <div className="space-y-0.5 overflow-y-auto scrollbar-thin scrollbar-thumb-[#27272a] scrollbar-track-transparent">
                {children}
            </div>
        </div>
    );
}

export function GeneralStatsCard({ game }: { game: GameDetail }) {
    return (
        <StatGroup title="General" awayLogo={game.away_logo} homeLogo={game.home_logo}>
            <StatRow label="Total Yards" awayValue={game.away_stats.total_yards} homeValue={game.home_stats.total_yards} />
            <StatRow label="Total Plays" awayValue={game.away_stats.plays} homeValue={game.home_stats.plays} />
            <StatRow label="Yards / Play" awayValue={game.away_stats.yards_per_play} homeValue={game.home_stats.yards_per_play} />
            <StatRow label="EPA / Play" awayValue={game.away_stats.epa_per_play} homeValue={game.home_stats.epa_per_play} format="epa" />
            <StatRow label="Passing Yards" awayValue={game.away_stats.passing_yards} homeValue={game.home_stats.passing_yards} isHeader />
            <StatRow label="Comp/Att" awayValue={game.away_stats.comp_att} homeValue={game.home_stats.comp_att} />
            <StatRow label="Yards / Pass" awayValue={game.away_stats.yards_per_pass} homeValue={game.home_stats.yards_per_pass} />
            <StatRow label="EPA / Pass" awayValue={game.away_stats.epa_per_pass} homeValue={game.home_stats.epa_per_pass} format="epa" />
            <StatRow label="Rushing Yards" awayValue={game.away_stats.rushing_yards} homeValue={game.home_stats.rushing_yards} isHeader />
            <StatRow label="Rush Attempts" awayValue={game.away_stats.rush_attempts} homeValue={game.home_stats.rush_attempts} />
            <StatRow label="Yards / Rush" awayValue={game.away_stats.yards_per_rush} homeValue={game.home_stats.yards_per_rush} />
            <StatRow label="EPA / Rush" awayValue={game.away_stats.epa_per_rush} homeValue={game.home_stats.epa_per_rush} format="epa" />
        </StatGroup>
    );
}

export function DownsConversionsCard({ game }: { game: GameDetail }) {
    // Parse 3rd down and red zone conversions
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
        <StatGroup title="Downs & Conversions" awayLogo={game.away_logo} homeLogo={game.home_logo}>
            <StatRow label="1st Downs" awayValue={game.away_stats.first_downs} homeValue={game.home_stats.first_downs} />
            <StatRow label="Passing 1st downs" awayValue={game.away_stats.passing_first_downs} homeValue={game.home_stats.passing_first_downs} />
            <StatRow label="Rushing 1st downs" awayValue={game.away_stats.rushing_first_downs} homeValue={game.home_stats.rushing_first_downs} />
            <StatRow label="Penalty 1st downs" awayValue={game.away_stats.penalty_first_downs} homeValue={game.home_stats.penalty_first_downs} />
            <StatRow label="3rd Down %" awayValue={away3rd.pct} homeValue={home3rd.pct} isHeader />
            <StatRow label="3rd Down Conversions" awayValue={away3rd.conv} homeValue={home3rd.conv} />
            <StatRow label="3rd Down Attempts" awayValue={away3rd.att} homeValue={home3rd.att} />
            <StatRow label="Red Zone %" awayValue={awayRZ.pct} homeValue={homeRZ.pct} isHeader />
            <StatRow label="Red Zone Conversions" awayValue={awayRZ.conv} homeValue={homeRZ.conv} />
            <StatRow label="Red Zone Attempts" awayValue={awayRZ.att} homeValue={homeRZ.att} />
        </StatGroup>
    );
}

export function NegativePlaysCard({ game }: { game: GameDetail }) {
    // Calculate turnover EPA if available (mock data doesn't have this, so using --)
    const awayTurnoverEPA = game.away_stats.turnovers ? "--" : "--";
    const homeTurnoverEPA = game.home_stats.turnovers ? "--" : "--";
    
    // Calculate sack EPA (mock data doesn't have this, so using --)
    const awaySackEPA = game.away_stats.sacks ? "--" : "--";
    const homeSackEPA = game.home_stats.sacks ? "--" : "--";
    
    // Calculate penalty EPA (mock data doesn't have this, so using --)
    const awayPenaltyEPA = game.away_stats.penalties ? "--" : "--";
    const homePenaltyEPA = game.home_stats.penalties ? "--" : "--";

    return (
        <StatGroup title="Negative Plays" awayLogo={game.away_logo} homeLogo={game.home_logo}>
            <StatRow label="Turnovers" awayValue={game.away_stats.turnovers || "--"} homeValue={game.home_stats.turnovers || "--"} />
            <StatRow label="Interceptions" awayValue="--" homeValue="--" />
            <StatRow label="Fumbles" awayValue="--" homeValue="--" />
            <StatRow label="Turnover EPA" awayValue={awayTurnoverEPA} homeValue={homeTurnoverEPA} format="epa" />
            <StatRow label="Sacks" awayValue={game.away_stats.sacks} homeValue={game.home_stats.sacks} isHeader />
            <StatRow label="Sack Yards" awayValue={game.away_stats.sack_yards} homeValue={game.home_stats.sack_yards} />
            <StatRow label="Sack EPA" awayValue={awaySackEPA} homeValue={homeSackEPA} format="epa" />
            <StatRow label="Penalties" awayValue={game.away_stats.penalties} homeValue={game.home_stats.penalties} isHeader />
            <StatRow label="Penalty Yards" awayValue={game.away_stats.penalty_yards} homeValue={game.home_stats.penalty_yards} />
            <StatRow label="Penalty EPA" awayValue={awayPenaltyEPA} homeValue={homePenaltyEPA} format="epa" />
        </StatGroup>
    );
}

// Keep TeamStats for backward compatibility, but it now returns all three cards
export function TeamStats({ game }: { game: GameDetail }) {
    return (
        <>
            <GeneralStatsCard game={game} />
            <DownsConversionsCard game={game} />
            <NegativePlaysCard game={game} />
        </>
    );
}

// Card-based stat row component for three-card layout
function CardStatRow({ 
    label, 
    awayValue, 
    homeValue, 
    format = "number", 
    isSectionHeader = false
}: {
    label: string;
    awayValue: string | number | undefined;
    homeValue: string | number | undefined;
    format?: "number" | "epa" | "percent";
    isSectionHeader?: boolean;
}) {
    const formatValue = (val: string | number | undefined) => {
        if (val === undefined || val === null || val === '--') return "--";
        if (format === "epa") {
            const num = Number(val);
            return (num >= 0 ? "+" : "") + num.toFixed(2);
        }
        if (format === "percent") {
            return typeof val === "string" && val.includes("%") ? val : `${val}%`;
        }
        return val;
    };

    const getColor = (val: string | number | undefined) => {
        if (format !== "epa" || val === undefined || val === '--') return "text-white";
        const num = Number(val);
        if (num === 0) return "text-white";
        return num > 0 ? "text-green-400" : "text-red-400";
    };

    return (
        <div className={cn(
            "flex justify-between py-1.5",
            isSectionHeader && "border-t border-[#27272a]/50 pt-3 mt-2"
        )}>
            <span className={cn(
                "text-xs",
                isSectionHeader ? "text-gray-300 font-semibold" : "text-gray-500 font-medium"
            )}>
                {label}
            </span>
            <div className="flex gap-6">
                <span className={cn("text-xs w-14 text-right font-mono font-semibold", getColor(awayValue))}>
                    {formatValue(awayValue)}
                </span>
                <span className={cn("text-xs w-14 text-right font-mono font-semibold", getColor(homeValue))}>
                    {formatValue(homeValue)}
                </span>
            </div>
        </div>
    );
}

// Card component for each section
function StatCard({ 
    title, 
    children, 
    awayLogo, 
    homeLogo 
}: { 
    title: string; 
    children: React.ReactNode; 
    awayLogo?: string; 
    homeLogo?: string;
}) {
    return (
        <div className="bg-[#18181b]/80 backdrop-blur-sm border border-[#27272a]/50 rounded-xl p-4 flex flex-col shadow-lg">
            {/* Header with team logos */}
            <div className="flex items-center justify-between mb-3 flex-shrink-0">
                <span className="text-sm text-gray-400 font-semibold">{title}</span>
                <div className="flex gap-3">
                    {awayLogo ? (
                        <img src={awayLogo} alt="Away" className="w-4 h-4 object-contain" />
                    ) : (
                        <div className="w-4 h-4 rounded-full bg-zinc-800 flex items-center justify-center text-[9px] text-zinc-400">A</div>
                    )}
                    {homeLogo ? (
                        <img src={homeLogo} alt="Home" className="w-4 h-4 object-contain" />
                    ) : (
                        <div className="w-4 h-4 rounded-full bg-zinc-800 flex items-center justify-center text-[9px] text-zinc-400">H</div>
                    )}
                </div>
            </div>
            <div className="space-y-0.5 overflow-y-auto scrollbar-thin scrollbar-thumb-[#27272a] scrollbar-track-transparent">
                {children}
            </div>
        </div>
    );
}

export function TeamStatsTable({ game }: { game: GameDetail }) {
    // Parse conversions
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

    // Parse comp/att with percentage
    const parseCompAtt = (val: string | undefined) => {
        if (!val) return { comp: "--", att: "--", pct: "--", display: "--" };
        const match = val.match(/(\d+)\/(\d+)/);
        if (match) {
            const comp = parseInt(match[1]);
            const att = parseInt(match[2]);
            const pct = att > 0 ? `${Math.round((comp / att) * 100)}%` : "--";
            return { comp, att, pct, display: `${comp}/${att} (${pct})` };
        }
        return { comp: val, att: "--", pct: "--", display: val };
    };

    const awayCompAtt = parseCompAtt(game.away_stats.comp_att);
    const homeCompAtt = parseCompAtt(game.home_stats.comp_att);

    return (
        <div className="w-full">
            <div className="text-xs text-gray-400 uppercase tracking-wider mb-4 font-semibold">
                Team Statistics
            </div>
            
            {/* Three Card Layout */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
                {/* Card 1: General */}
                <StatCard title="General" awayLogo={game.away_logo} homeLogo={game.home_logo}>
                    <CardStatRow label="Total Yards" awayValue={game.away_stats.total_yards} homeValue={game.home_stats.total_yards} />
                    <CardStatRow label="Total Plays" awayValue={game.away_stats.plays} homeValue={game.home_stats.plays} />
                    <CardStatRow label="Yards / Play" awayValue={game.away_stats.yards_per_play} homeValue={game.home_stats.yards_per_play} />
                    <CardStatRow label="EPA / Play" awayValue={game.away_stats.epa_per_play} homeValue={game.home_stats.epa_per_play} format="epa" />
                    <CardStatRow label="Passing Yards" awayValue={game.away_stats.passing_yards} homeValue={game.home_stats.passing_yards} isSectionHeader={true} />
                    <CardStatRow label="Comp/Att" awayValue={awayCompAtt.display} homeValue={homeCompAtt.display} />
                    <CardStatRow label="Yards / Pass" awayValue={game.away_stats.yards_per_pass} homeValue={game.home_stats.yards_per_pass} />
                    <CardStatRow label="EPA / Pass" awayValue={game.away_stats.epa_per_pass} homeValue={game.home_stats.epa_per_pass} format="epa" />
                    <CardStatRow label="Rushing Yards" awayValue={game.away_stats.rushing_yards} homeValue={game.home_stats.rushing_yards} isSectionHeader={true} />
                    <CardStatRow label="Rush Attempts" awayValue={game.away_stats.rush_attempts} homeValue={game.home_stats.rush_attempts} />
                    <CardStatRow label="Yards / Rush" awayValue={game.away_stats.yards_per_rush} homeValue={game.home_stats.yards_per_rush} />
                    <CardStatRow label="EPA / Rush" awayValue={game.away_stats.epa_per_rush} homeValue={game.home_stats.epa_per_rush} format="epa" />
                </StatCard>

                {/* Card 2: Downs & Conversions */}
                <StatCard title="Downs & Conversions" awayLogo={game.away_logo} homeLogo={game.home_logo}>
                    <CardStatRow label="1st Downs" awayValue={game.away_stats.first_downs} homeValue={game.home_stats.first_downs} />
                    <CardStatRow label="Passing 1st downs" awayValue={game.away_stats.passing_first_downs} homeValue={game.home_stats.passing_first_downs} />
                    <CardStatRow label="Rushing 1st downs" awayValue={game.away_stats.rushing_first_downs} homeValue={game.home_stats.rushing_first_downs} />
                    <CardStatRow label="Penalty 1st downs" awayValue={game.away_stats.penalty_first_downs || "--"} homeValue={game.home_stats.penalty_first_downs || "--"} />
                    <CardStatRow label="3rd Down %" awayValue={away3rd.pct} homeValue={home3rd.pct} format="percent" isSectionHeader={true} />
                    <CardStatRow label="3rd Down Conversions" awayValue={away3rd.conv} homeValue={home3rd.conv} />
                    <CardStatRow label="3rd Down Attempts" awayValue={away3rd.att} homeValue={home3rd.att} />
                    <CardStatRow label="Red Zone %" awayValue={awayRZ.pct} homeValue={homeRZ.pct} format="percent" isSectionHeader={true} />
                    <CardStatRow label="Red Zone Conversions" awayValue={awayRZ.conv} homeValue={homeRZ.conv} />
                    <CardStatRow label="Red Zone Attempts" awayValue={awayRZ.att} homeValue={homeRZ.att} />
                </StatCard>

                {/* Card 3: Negative Plays */}
                <StatCard title="Negative Plays" awayLogo={game.away_logo} homeLogo={game.home_logo}>
                    <CardStatRow label="Turnovers" awayValue={game.away_stats.turnovers || "--"} homeValue={game.home_stats.turnovers || "--"} />
                    <CardStatRow label="Interceptions" awayValue="--" homeValue="--" />
                    <CardStatRow label="Fumbles" awayValue="--" homeValue="--" />
                    <CardStatRow label="Turnover EPA" awayValue="--" homeValue="--" format="epa" />
                    <CardStatRow label="Sacks" awayValue={game.away_stats.sacks} homeValue={game.home_stats.sacks} isSectionHeader={true} />
                    <CardStatRow label="Sack Yards" awayValue={game.away_stats.sack_yards} homeValue={game.home_stats.sack_yards} />
                    <CardStatRow label="Sack EPA" awayValue="--" homeValue="--" format="epa" />
                    <CardStatRow label="Penalties" awayValue={game.away_stats.penalties || "--"} homeValue={game.home_stats.penalties || "--"} isSectionHeader={true} />
                    <CardStatRow label="Penalty Yards" awayValue={game.away_stats.penalty_yards || "--"} homeValue={game.home_stats.penalty_yards || "--"} />
                    <CardStatRow label="Penalty EPA" awayValue="--" homeValue="--" format="epa" />
                </StatCard>
            </div>
        </div>
    );
}
