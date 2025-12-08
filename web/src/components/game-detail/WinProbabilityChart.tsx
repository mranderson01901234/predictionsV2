"use client";

import { GameDetail } from "@/lib/mock_data";
import {
    AreaChart,
    Area,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    ReferenceLine,
} from "recharts";

interface WinProbabilityChartProps {
    game: GameDetail;
}

export function WinProbabilityChart({ game }: WinProbabilityChartProps) {
    const data = game.win_probability || [];
    
    // Transform data to use percentage (0-100) and map time to quarter labels
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
            label = "1st";
        }
        
        return {
            time: label,
            probability: item.home_prob * 100, // Convert to percentage
        };
    });

    // Get unique quarter labels for X-axis
    const quarterLabels = ["1st", "2nd", "Half", "3rd", "End"];

    return (
        <div className="h-full w-full bg-[#18181b]/80 backdrop-blur-sm border border-[#27272a]/50 rounded-xl p-5 flex flex-col shadow-lg">
            {/* Header */}
            <div className="text-xs text-gray-400 uppercase tracking-wider mb-4 font-semibold">
                Win Probability
            </div>

            {/* Chart Container */}
            <div className="h-[220px] relative flex-1 min-h-[220px] -mx-1">
                {/* Team logos at probability endpoints */}
                <div className="absolute left-3 top-3 z-10">
                    {game.away_logo ? (
                        <div className="w-7 h-7 rounded-full bg-[#18181b] border border-[#27272a] p-0.5 flex items-center justify-center">
                            <img src={game.away_logo} alt={game.away_team} className="w-full h-full object-contain rounded-full" />
                        </div>
                    ) : (
                        <div className="w-7 h-7 rounded-full bg-zinc-800 border border-[#27272a] flex items-center justify-center text-[10px] text-zinc-400">
                            {game.away_team[0]}
                        </div>
                    )}
                </div>
                <div className="absolute left-3 bottom-3 z-10">
                    {game.home_logo ? (
                        <div className="w-7 h-7 rounded-full bg-[#18181b] border border-[#27272a] p-0.5 flex items-center justify-center">
                            <img src={game.home_logo} alt={game.home_team} className="w-full h-full object-contain rounded-full" />
                        </div>
                    ) : (
                        <div className="w-7 h-7 rounded-full bg-zinc-800 border border-[#27272a] flex items-center justify-center text-[10px] text-zinc-400">
                            {game.home_team[0]}
                        </div>
                    )}
                </div>

                {/* Recharts Area Chart */}
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={chartData} margin={{ top: 15, right: 15, left: 40, bottom: 25 }}>
                        <defs>
                            <linearGradient id="winProbGradient" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.4} />
                                <stop offset="50%" stopColor="#3b82f6" stopOpacity={0.2} />
                                <stop offset="100%" stopColor="#3b82f6" stopOpacity={0} />
                            </linearGradient>
                        </defs>
                        <XAxis
                            dataKey="time"
                            tick={{ fill: '#9ca3af', fontSize: 11, fontWeight: 500 }}
                            axisLine={false}
                            tickLine={false}
                            tickMargin={8}
                        />
                        <YAxis
                            domain={[0, 100]}
                            tick={{ fill: '#9ca3af', fontSize: 11, fontWeight: 500 }}
                            axisLine={false}
                            tickLine={false}
                            tickMargin={8}
                            tickFormatter={(v) => `${v}%`}
                        />
                        <ReferenceLine y={50} stroke="#4b5563" strokeDasharray="4 4" strokeWidth={1.5} opacity={0.6} />
                        <Area
                            type="monotone"
                            dataKey="probability"
                            stroke="#3b82f6"
                            fill="url(#winProbGradient)"
                            strokeWidth={2.5}
                            dot={false}
                            activeDot={{ r: 4, fill: '#3b82f6', stroke: '#18181b', strokeWidth: 2 }}
                        />
                        <Tooltip
                            contentStyle={{
                                backgroundColor: '#18181b',
                                border: '1px solid #27272a',
                                borderRadius: '8px',
                                padding: '8px 12px',
                                boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.3)'
                            }}
                            labelStyle={{ color: '#9ca3af', fontSize: '11px', marginBottom: '4px' }}
                            itemStyle={{ color: '#3b82f6', fontSize: '13px', fontWeight: 600 }}
                            formatter={(value: number) => [`${value.toFixed(1)}%`, 'Home Win Prob']}
                        />
                    </AreaChart>
                </ResponsiveContainer>
            </div>

            {/* X-axis labels */}
            <div className="flex justify-between text-xs text-gray-500 mt-3 px-1">
                {quarterLabels.map((label) => (
                    <span key={label} className="font-medium">{label}</span>
                ))}
            </div>
        </div>
    );
}
