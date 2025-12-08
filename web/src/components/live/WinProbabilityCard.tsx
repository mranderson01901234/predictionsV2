"use client";

import { GameDetail } from "@/lib/mock_data";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
import { TrendingUp } from "lucide-react";
import {
    AreaChart,
    Area,
    XAxis,
    YAxis,
    Tooltip,
    ResponsiveContainer,
    ReferenceLine,
} from "recharts";

interface WinProbabilityCardProps {
    game: GameDetail;
}

export function WinProbabilityCard({ game }: WinProbabilityCardProps) {
    const data = game.win_probability || [];
    
    // Transform data for the chart
    const chartData = data.map((item, index) => {
        // Map time strings to quarter labels
        let label = "1st";
        if (item.time.includes("Q2") || item.time.includes("2nd")) {
            label = index === data.length - 1 ? "Half" : "2nd";
        } else if (item.time.includes("Q3") || item.time.includes("3rd")) {
            label = "3rd";
        } else if (item.time.includes("Q4") || item.time.includes("4th") || item.time.includes("End")) {
            label = "End";
        } else if (item.time.includes("Pre")) {
            label = "Pre";
        }
        
        return {
            time: label,
            rawTime: item.time,
            homeProb: item.home_prob * 100,
            awayProb: (1 - item.home_prob) * 100,
        };
    });

    // Get current win probability for display
    const currentProb = chartData.length > 0 
        ? chartData[chartData.length - 1].homeProb 
        : 50;
    
    const homeLeading = currentProb > 50;
    const probChange = chartData.length > 1 
        ? currentProb - chartData[chartData.length - 2].homeProb
        : 0;

    // Custom tooltip
    const CustomTooltip = ({ active, payload, label }: any) => {
        if (active && payload && payload.length) {
            const prob = payload[0].value;
            return (
                <div className="glass-surface-elevated px-3 py-2 border border-[var(--glass-border-medium)]">
                    <p className="text-[10px] text-[var(--muted-foreground)] mb-1">{label}</p>
                    <div className="flex items-baseline gap-1.5">
                        <span className="text-lg font-bold font-mono text-[var(--neon-blue)]">
                            {prob.toFixed(1)}%
                        </span>
                        <span className="text-[10px] text-[var(--muted-foreground)]">
                            {game.home_team}
                        </span>
                    </div>
                </div>
            );
        }
        return null;
    };

    // Custom dot for the current point
    const CustomActiveDot = (props: any) => {
        const { cx, cy, payload, index } = props;
        const isLast = index === chartData.length - 1;
        
        if (!isLast) return null;
        
        return (
            <g>
                {/* Outer glow */}
                <circle
                    cx={cx}
                    cy={cy}
                    r={12}
                    fill="url(#dotGlow)"
                    opacity={0.6}
                />
                {/* Inner dot */}
                <circle
                    cx={cx}
                    cy={cy}
                    r={5}
                    fill="var(--neon-blue)"
                    stroke="var(--background)"
                    strokeWidth={2}
                />
            </g>
        );
    };

    return (
        <motion.div
            className="glass-surface-elevated h-full flex flex-col p-4"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
        >
            {/* Header */}
            <div className="flex items-start justify-between mb-4">
                <div className="flex items-center gap-3">
                    <div className="p-2 rounded-lg bg-[var(--glass-bg-elevated)] border border-[var(--glass-border)]">
                        <TrendingUp size={16} className="text-[var(--neon-blue)]" />
                    </div>
                    <div>
                        <h3 className="text-sm font-semibold text-[var(--foreground)]">
                            Win Probability
                        </h3>
                        <p className="text-[10px] text-[var(--muted-foreground)]">
                            Live model prediction
                        </p>
                    </div>
                </div>
                
                {/* Team Legend */}
                <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2">
                        {game.home_logo ? (
                            <img 
                                src={game.home_logo} 
                                alt={game.home_team} 
                                className="w-5 h-5 object-contain" 
                            />
                        ) : (
                            <div className="w-5 h-5 rounded-full bg-[var(--neon-blue-muted)] border border-[var(--neon-blue)]/30 flex items-center justify-center text-[8px] font-bold text-[var(--neon-blue)]">
                                {game.home_team[0]}
                            </div>
                        )}
                        <span className="text-[10px] text-[var(--foreground)] font-medium">
                            {game.home_team}
                        </span>
                    </div>
                    <div className="flex items-center gap-2">
                        {game.away_logo ? (
                            <img 
                                src={game.away_logo} 
                                alt={game.away_team} 
                                className="w-5 h-5 object-contain opacity-60" 
                            />
                        ) : (
                            <div className="w-5 h-5 rounded-full bg-[var(--glass-bg)] border border-[var(--glass-border)] flex items-center justify-center text-[8px] text-[var(--muted-foreground)]">
                                {game.away_team[0]}
                            </div>
                        )}
                        <span className="text-[10px] text-[var(--muted-foreground)]">
                            {game.away_team}
                        </span>
                    </div>
                </div>
            </div>

            {/* Chart Container */}
            <div className="flex-1 min-h-[200px] h-[200px] relative">
                {/* Y-axis Team indicators */}
                <div className="absolute left-0 top-0 z-10">
                    {game.away_logo ? (
                        <div className="w-6 h-6 rounded-lg bg-[var(--glass-bg-elevated)] border border-[var(--glass-border)] p-0.5 flex items-center justify-center">
                            <img 
                                src={game.away_logo} 
                                alt={game.away_team} 
                                className="w-full h-full object-contain opacity-60" 
                            />
                        </div>
                    ) : (
                        <div className="w-6 h-6 rounded-lg bg-[var(--glass-bg-elevated)] border border-[var(--glass-border)] flex items-center justify-center text-[9px] text-[var(--muted-foreground)]">
                            {game.away_team[0]}
                        </div>
                    )}
                </div>
                <div className="absolute left-0 bottom-6 z-10">
                    {game.home_logo ? (
                        <div className="w-6 h-6 rounded-lg bg-[var(--neon-blue-muted)] border border-[var(--neon-blue)]/30 p-0.5 flex items-center justify-center">
                            <img 
                                src={game.home_logo} 
                                alt={game.home_team} 
                                className="w-full h-full object-contain" 
                            />
                        </div>
                    ) : (
                        <div className="w-6 h-6 rounded-lg bg-[var(--neon-blue-muted)] border border-[var(--neon-blue)]/30 flex items-center justify-center text-[9px] font-bold text-[var(--neon-blue)]">
                            {game.home_team[0]}
                        </div>
                    )}
                </div>

                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart 
                        data={chartData} 
                        margin={{ top: 10, right: 10, left: 35, bottom: 20 }}
                    >
                        <defs>
                            {/* Premium gradient fill */}
                            <linearGradient id="winProbGradient" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="0%" stopColor="var(--neon-blue)" stopOpacity={0.4} />
                                <stop offset="50%" stopColor="var(--neon-blue)" stopOpacity={0.15} />
                                <stop offset="100%" stopColor="var(--neon-blue)" stopOpacity={0} />
                            </linearGradient>
                            
                            {/* Glow for current point */}
                            <radialGradient id="dotGlow" cx="50%" cy="50%" r="50%">
                                <stop offset="0%" stopColor="var(--neon-blue)" stopOpacity={0.6} />
                                <stop offset="100%" stopColor="var(--neon-blue)" stopOpacity={0} />
                            </radialGradient>
                            
                            {/* Line glow filter */}
                            <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
                                <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
                                <feMerge>
                                    <feMergeNode in="coloredBlur"/>
                                    <feMergeNode in="SourceGraphic"/>
                                </feMerge>
                            </filter>
                        </defs>
                        
                        <XAxis
                            dataKey="time"
                            tick={{ fill: 'var(--muted-foreground)', fontSize: 10, fontWeight: 500 }}
                            axisLine={false}
                            tickLine={false}
                            tickMargin={8}
                        />
                        
                        <YAxis
                            domain={[0, 100]}
                            tick={{ fill: 'var(--muted-foreground)', fontSize: 10, fontWeight: 500 }}
                            axisLine={false}
                            tickLine={false}
                            tickMargin={8}
                            tickFormatter={(v) => `${v}%`}
                            ticks={[0, 25, 50, 75, 100]}
                        />
                        
                        {/* 50% reference line */}
                        <ReferenceLine 
                            y={50} 
                            stroke="var(--glass-border-strong)" 
                            strokeDasharray="4 4" 
                            strokeWidth={1}
                        />
                        
                        {/* Confidence band (optional visual enhancement) */}
                        <Area
                            type="monotone"
                            dataKey="homeProb"
                            stroke="none"
                            fill="url(#winProbGradient)"
                            fillOpacity={0.3}
                        />
                        
                        {/* Main probability line */}
                        <Area
                            type="monotone"
                            dataKey="homeProb"
                            stroke="var(--neon-blue)"
                            fill="url(#winProbGradient)"
                            strokeWidth={2.5}
                            dot={false}
                            activeDot={<CustomActiveDot />}
                            style={{ filter: 'url(#glow)' }}
                        />
                        
                        <Tooltip 
                            content={<CustomTooltip />} 
                            cursor={{ stroke: 'var(--glass-border-medium)', strokeDasharray: '4 4' }}
                        />
                    </AreaChart>
                </ResponsiveContainer>
            </div>

            {/* Current probability footer */}
            <div className="pt-4 mt-2 border-t border-[var(--glass-border)] flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <span className="text-[10px] text-[var(--muted-foreground)] uppercase tracking-wide font-medium">
                        Current Win Prob
                    </span>
                </div>
                <div className="flex items-center gap-3">
                    {/* Change indicator */}
                    {probChange !== 0 && (
                        <span className={cn(
                            "text-[10px] font-mono font-semibold flex items-center gap-0.5",
                            probChange > 0 ? "text-[var(--neon-green)]" : "text-[var(--destructive)]"
                        )}>
                            {probChange > 0 ? "↑" : "↓"}
                            {Math.abs(probChange).toFixed(1)}%
                        </span>
                    )}
                    
                    {/* Main value */}
                    <div className="flex items-center gap-2">
                        <span className={cn(
                            "text-xl font-bold font-mono",
                            homeLeading ? "text-neon-blue" : "text-[var(--muted-foreground)]"
                        )}>
                            {currentProb.toFixed(0)}%
                        </span>
                        <span className="text-[10px] text-[var(--muted-foreground)] font-medium">
                            {game.home_team}
                        </span>
                    </div>
                </div>
            </div>
        </motion.div>
    );
}

export default WinProbabilityCard;
