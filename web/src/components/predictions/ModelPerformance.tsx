"use client";

import { cn } from "@/lib/utils";
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    BarChart,
    Bar,
    ReferenceLine
} from "recharts";

// Mock Data for Performance
const PERFORMANCE_STATS = {
    accuracy: "69.1%",
    atsWinRate: "56.89%",
    mae: "10.1",
    clv: "5.75%"
};

const CUMULATIVE_DATA = [
    { year: '2018', units: 0 },
    { year: '2019', units: 12.5 },
    { year: '2020', units: 25.3 },
    { year: '2021', units: 18.2 },
    { year: '2022', units: 35.6 },
    { year: '2023', units: 42.1 },
    { year: '2024', units: 58.4 },
    { year: '2025', units: 72.3 },
];

const SEASON_DATA = [
    { season: '2025', modelSu: '69.1%', vsClose: '+4.5%', mae: 9.8, units: 13.9, clv: '5.2%' },
    { season: '2024', modelSu: '66.2%', vsClose: '+1.2%', mae: 10.2, units: 16.3, clv: '4.8%' },
    { season: '2023', modelSu: '65.8%', vsClose: '+0.8%', mae: 10.5, units: 6.5, clv: '3.9%' },
    { season: '2022', modelSu: '67.4%', vsClose: '+2.1%', mae: 10.1, units: 17.4, clv: '4.5%' },
];

function StatCard({ title, value, sub, sub2 }: { title: string, value: string, sub: string, sub2?: string }) {
    return (
        <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-xl p-4">
            <h3 className="text-[#a0a0a0] text-xs uppercase tracking-wider mb-2">{title}</h3>
            <div className="text-2xl font-bold text-white mb-1">{value}</div>
            <div className="text-xs text-[#666666]">{sub}</div>
            {sub2 && <div className="text-xs text-[#666666]">{sub2}</div>}
        </div>
    );
}

export function ModelPerformance() {
    return (
        <div className="max-w-6xl mx-auto p-6 space-y-8">
            <div className="flex justify-between items-center">
                <div>
                    <h1 className="text-2xl font-bold text-white">Model Performance</h1>
                    <p className="text-[#a0a0a0]">Historical performance metrics and validation</p>
                </div>
                <div className="flex bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg p-1">
                    <button className="px-3 py-1 text-xs font-medium bg-[#242424] text-white rounded">All Time</button>
                    <button className="px-3 py-1 text-xs font-medium text-[#666666] hover:text-white">Current Season</button>
                </div>
            </div>

            {/* Summary Stats */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <StatCard
                    title="Accuracy"
                    value={PERFORMANCE_STATS.accuracy}
                    sub="% of winners predicted"
                    sub2="vs Open +0.33"
                />
                <StatCard
                    title="ATS Win Rate"
                    value={PERFORMANCE_STATS.atsWinRate}
                    sub="vs opening line"
                    sub2="vs Close 53.75%"
                />
                <StatCard
                    title="Predict Error"
                    value={PERFORMANCE_STATS.mae}
                    sub="Model MAE"
                    sub2="vs Open +0.10"
                />
                <StatCard
                    title="CLV"
                    value={PERFORMANCE_STATS.clv}
                    sub="Avg Closing Line Value"
                    sub2="Exp Units 72.3"
                />
            </div>

            {/* Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-2 bg-[#1a1a1a] border border-[#2a2a2a] rounded-xl p-6">
                    <h3 className="text-white font-bold mb-4">Cumulative Units Won</h3>
                    <div className="h-[300px]">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={CUMULATIVE_DATA}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#2a2a2a" vertical={false} />
                                <XAxis dataKey="year" stroke="#666666" tick={{ fontSize: 12 }} axisLine={false} tickLine={false} />
                                <YAxis stroke="#666666" tick={{ fontSize: 12 }} axisLine={false} tickLine={false} />
                                <Tooltip
                                    contentStyle={{ backgroundColor: '#1a1a1a', borderColor: '#2a2a2a', color: '#fff' }}
                                    itemStyle={{ color: '#3b82f6' }}
                                />
                                <Line type="monotone" dataKey="units" stroke="#3b82f6" strokeWidth={3} dot={{ r: 4, fill: "#3b82f6" }} />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-xl p-6">
                    <h3 className="text-white font-bold mb-4">Units by Season</h3>
                    <div className="h-[300px]">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={SEASON_DATA}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#2a2a2a" vertical={false} />
                                <XAxis dataKey="season" stroke="#666666" tick={{ fontSize: 12 }} axisLine={false} tickLine={false} />
                                <YAxis stroke="#666666" tick={{ fontSize: 12 }} axisLine={false} tickLine={false} />
                                <Tooltip
                                    cursor={{ fill: '#242424' }}
                                    contentStyle={{ backgroundColor: '#1a1a1a', borderColor: '#2a2a2a', color: '#fff' }}
                                />
                                <ReferenceLine y={0} stroke="#666666" />
                                <Bar dataKey="units" fill="#22c55e" radius={[4, 4, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>

            {/* Season Detail Table */}
            <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-xl overflow-hidden">
                <div className="p-4 border-b border-[#2a2a2a]">
                    <h3 className="text-white font-bold">Season Detail</h3>
                </div>
                <div className="overflow-x-auto">
                    <table className="w-full text-sm text-left">
                        <thead className="text-xs text-[#666666] uppercase bg-[#242424]">
                            <tr>
                                <th className="px-6 py-3">Season</th>
                                <th className="px-6 py-3">Model SU%</th>
                                <th className="px-6 py-3">vs Close</th>
                                <th className="px-6 py-3">Model MAE</th>
                                <th className="px-6 py-3">Units</th>
                                <th className="px-6 py-3">CLV</th>
                            </tr>
                        </thead>
                        <tbody>
                            {SEASON_DATA.map((season, i) => (
                                <tr key={season.season} className="border-b border-[#2a2a2a] hover:bg-[#242424] transition-colors">
                                    <td className="px-6 py-4 font-medium text-white">{season.season}</td>
                                    <td className="px-6 py-4 text-[#a0a0a0]">{season.modelSu}</td>
                                    <td className="px-6 py-4 text-[#22c55e]">{season.vsClose}</td>
                                    <td className="px-6 py-4 text-[#a0a0a0]">{season.mae}</td>
                                    <td className="px-6 py-4 font-bold text-[#22c55e]">+{season.units}</td>
                                    <td className="px-6 py-4 text-[#a0a0a0]">{season.clv}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
}
