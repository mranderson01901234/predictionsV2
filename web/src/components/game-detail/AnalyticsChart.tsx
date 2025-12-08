"use client";

import { GlassCard } from "@/components/ui/glass";
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    ReferenceLine,
} from "recharts";

interface ChartDataPoint {
    time: string;
    market: number;
    model: number;
}

// Mock historical data generator
const generateMockHistory = (startSpread: number, endSpread: number, modelSpread: number) => {
    const data: ChartDataPoint[] = [];
    const points = 10;
    for (let i = 0; i < points; i++) {
        const progress = i / (points - 1);
        // Linear interpolation with some noise
        const currentMarket = startSpread + (endSpread - startSpread) * progress + (Math.random() * 0.5 - 0.25);
        data.push({
            time: `Day ${i - points + 1}`,
            market: Number(currentMarket.toFixed(1)),
            model: modelSpread, // Model is usually static or updates less frequently
        });
    }
    return data;
};

interface AnalyticsChartProps {
    marketSpread: number;
    modelSpread: number;
}

export function AnalyticsChart({ marketSpread, modelSpread }: AnalyticsChartProps) {
    // Simulate opening line being 2 points different
    const openSpread = marketSpread + (Math.random() > 0.5 ? 1.5 : -1.5);
    const data = generateMockHistory(openSpread, marketSpread, modelSpread);

    return (
        <GlassCard className="h-[400px] w-full p-4">
            <div className="mb-4">
                <h3 className="text-lg font-semibold text-white">Line Movement vs Model</h3>
                <p className="text-sm text-slate-400">Tracking spread value over time</p>
            </div>

            <ResponsiveContainer width="100%" height="85%">
                <LineChart data={data}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                    <XAxis
                        dataKey="time"
                        stroke="#64748b"
                        tick={{ fontSize: 12 }}
                        tickLine={false}
                        axisLine={false}
                    />
                    <YAxis
                        stroke="#64748b"
                        tick={{ fontSize: 12 }}
                        tickLine={false}
                        axisLine={false}
                        domain={['auto', 'auto']}
                    />
                    <Tooltip
                        contentStyle={{
                            backgroundColor: "#0f172a",
                            borderColor: "#1e293b",
                            borderRadius: "8px",
                            color: "#f8fafc",
                        }}
                        itemStyle={{ color: "#f8fafc" }}
                    />
                    <ReferenceLine y={0} stroke="#475569" strokeDasharray="3 3" />

                    <Line
                        type="monotone"
                        dataKey="market"
                        name="Market Spread"
                        stroke="#60a5fa" // Blue
                        strokeWidth={2}
                        dot={{ r: 4, fill: "#60a5fa" }}
                        activeDot={{ r: 6 }}
                    />
                    <Line
                        type="step"
                        dataKey="model"
                        name="Model Projection"
                        stroke="#34d399" // Emerald
                        strokeWidth={2}
                        strokeDasharray="5 5"
                        dot={false}
                    />
                </LineChart>
            </ResponsiveContainer>
        </GlassCard>
    );
}
