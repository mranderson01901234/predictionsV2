"use client";

import { Prediction } from "@/lib/mock_data";
import { GlassCard } from "@/components/ui/glass";
import { cn } from "@/lib/utils";
import { TrendingUp, AlertTriangle, CheckCircle2, XCircle } from "lucide-react";

interface PredictionCardProps {
    prediction: Prediction;
    marketSpread: number;
}

export function PredictionCard({ prediction, marketSpread }: PredictionCardProps) {
    const edge = prediction.edge_spread;
    const isGoodBet = Math.abs(edge) > 1.5;
    const confidenceColor = prediction.confidence_score > 70 ? "text-emerald-400" : "text-yellow-400";

    // Determine recommendation text
    let recommendation = "NO BET";
    let recColor = "text-slate-400";

    if (isGoodBet) {
        if (edge > 0) {
            recommendation = "BET HOME";
            recColor = "text-emerald-400";
        } else {
            recommendation = "BET AWAY";
            recColor = "text-emerald-400";
        }
    }

    return (
        <GlassCard className="h-full flex flex-col justify-between relative overflow-hidden">
            {/* Background Glow */}
            <div className={cn(
                "absolute -top-20 -right-20 w-40 h-40 rounded-full blur-3xl opacity-20",
                isGoodBet ? "bg-emerald-500" : "bg-slate-500"
            )} />

            <div>
                <h3 className="text-lg font-semibold text-white mb-1">Model Verdict</h3>
                <div className="flex items-center gap-2 mb-6">
                    <span className={cn("text-3xl font-black tracking-tight", recColor)}>
                        {recommendation}
                    </span>
                    {isGoodBet && <CheckCircle2 className="w-6 h-6 text-emerald-400" />}
                </div>

                <div className="space-y-4">
                    {/* Confidence Meter */}
                    <div>
                        <div className="flex justify-between text-sm mb-1">
                            <span className="text-slate-400">Confidence</span>
                            <span className={cn("font-bold", confidenceColor)}>{prediction.confidence_score}%</span>
                        </div>
                        <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                            <div
                                className={cn("h-full rounded-full transition-all duration-1000",
                                    prediction.confidence_score > 70 ? "bg-emerald-500" : "bg-yellow-500"
                                )}
                                style={{ width: `${prediction.confidence_score}%` }}
                            />
                        </div>
                    </div>

                    {/* Edge Analysis */}
                    <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/50">
                        <div className="flex justify-between items-center mb-2">
                            <span className="text-sm text-slate-400">Market Line</span>
                            <span className="font-mono text-white">{marketSpread > 0 ? "+" : ""}{marketSpread}</span>
                        </div>
                        <div className="flex justify-between items-center mb-2">
                            <span className="text-sm text-slate-400">Model Line</span>
                            <span className="font-mono text-blue-400">{prediction.predicted_spread > 0 ? "+" : ""}{prediction.predicted_spread.toFixed(1)}</span>
                        </div>
                        <div className="flex justify-between items-center pt-2 border-t border-slate-700">
                            <span className="text-sm font-bold text-white">Edge</span>
                            <span className={cn("font-mono font-bold", Math.abs(edge) > 1.0 ? "text-emerald-400" : "text-slate-400")}>
                                {edge > 0 ? "+" : ""}{edge.toFixed(1)} pts
                            </span>
                        </div>
                    </div>
                </div>
            </div>

            <div className="mt-6 text-xs text-slate-500 text-center">
                Model updated: {new Date().toLocaleTimeString()}
            </div>
        </GlassCard>
    );
}
