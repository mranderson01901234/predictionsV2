"use client";

import { Quarterback } from "@/lib/mock_data";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
import { User } from "lucide-react";
import { useState } from "react";

interface QuarterbackPerformanceCardProps {
    homeQB: Quarterback;
    awayQB: Quarterback;
}

// Two separate QB cards stacked vertically
export function QuarterbackPerformanceCard({ homeQB, awayQB }: QuarterbackPerformanceCardProps) {
    return (
        <div className="flex flex-col gap-3 h-full">
            {/* Away QB Card */}
            <QBCard qb={awayQB} isHome={false} />
            {/* Home QB Card */}
            <QBCard qb={homeQB} isHome={true} />
        </div>
    );
}

interface QBCardProps {
    qb: Quarterback;
    isHome: boolean;
}

function QBCard({ qb, isHome }: QBCardProps) {
    const [imageError, setImageError] = useState(false);
    const epaPerPlay = qb.epa / (qb.attempts || 1);
    const completionPct = qb.attempts > 0
        ? Math.round((qb.completions / qb.attempts) * 100)
        : 0;

    const showFallback = !qb.headshot_url || imageError;

    return (
        <motion.div
            className="glass-surface flex-1 flex flex-col overflow-hidden min-h-[200px]"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: isHome ? 0.2 : 0.1 }}
        >
            {/* Header Row with Large Headshot */}
            <div className="p-4 pb-3 flex items-center gap-4 border-b border-[var(--glass-border)]">
                {/* Large Circular Headshot */}
                <div className={cn(
                    "w-20 h-20 rounded-full bg-[var(--glass-bg-elevated)] flex items-center justify-center overflow-hidden flex-shrink-0 border-2",
                    isHome ? "border-[var(--neon-blue)]/30" : "border-[var(--neon-purple)]/30"
                )}>
                    {!showFallback && qb.headshot_url ? (
                        <img
                            src={qb.headshot_url}
                            alt={qb.name}
                            className="w-full h-full object-cover"
                            onError={() => setImageError(true)}
                        />
                    ) : (
                        <User size={40} className="text-white/20" />
                    )}
                </div>

                {/* Name & Team */}
                <div className="flex-1 min-w-0">
                    <div className="font-semibold text-base text-[var(--foreground)] truncate mb-0.5">{qb.name}</div>
                    <div className="text-xs text-[var(--muted-foreground)]">{qb.team} • QB</div>
                </div>

                {/* QBR Badge */}
                {qb.qbr !== undefined && (
                    <div className="text-right flex-shrink-0">
                        <div className={cn(
                            "text-2xl font-bold font-mono",
                            qb.qbr >= 70 ? "text-[var(--neon-green)]" :
                            qb.qbr >= 50 ? "text-[var(--foreground)]" : "text-[var(--warning)]"
                        )}>
                            {qb.qbr.toFixed(1)}
                        </div>
                        <div className="text-[10px] text-[var(--muted-foreground)] uppercase">QBR</div>
                    </div>
                )}
            </div>

            {/* Player Info & Stats */}
            <div className="p-4 flex-1 flex flex-col min-h-0">

                {/* Stats Grid */}
                <div className="grid grid-cols-4 gap-1 text-center mb-3 flex-shrink-0">
                    <div className="p-1.5 rounded-lg bg-[var(--glass-bg-subtle)]">
                        <div className="text-[8px] text-[var(--muted-foreground)] uppercase font-medium">Pass</div>
                        <div className="text-[11px] text-[var(--foreground)] font-mono font-semibold">{qb.completions}/{qb.attempts}</div>
                    </div>
                    <div className="p-1.5 rounded-lg bg-[var(--glass-bg-subtle)]">
                        <div className="text-[8px] text-[var(--muted-foreground)] uppercase font-medium">Yds</div>
                        <div className="text-[11px] text-[var(--foreground)] font-mono font-semibold">{qb.yards}</div>
                    </div>
                    <div className="p-1.5 rounded-lg bg-[var(--glass-bg-subtle)]">
                        <div className="text-[8px] text-[var(--muted-foreground)] uppercase font-medium">TD</div>
                        <div className="text-[11px] text-[var(--neon-green)] font-mono font-semibold">{qb.tds}</div>
                    </div>
                    <div className="p-1.5 rounded-lg bg-[var(--glass-bg-subtle)]">
                        <div className="text-[8px] text-[var(--muted-foreground)] uppercase font-medium">INT</div>
                        <div className={cn(
                            "text-[11px] font-mono font-semibold",
                            qb.ints > 0 ? "text-[var(--destructive)]" : "text-[var(--foreground)]"
                        )}>
                            {qb.ints}
                        </div>
                    </div>
                </div>

                {/* EPA & Completion % */}
                <div className="mt-auto pt-2 border-t border-[var(--glass-border)] flex justify-between items-center flex-shrink-0">
                    <div className="flex items-baseline gap-1">
                        <span className="text-[9px] text-[var(--muted-foreground)] font-medium">EPA/Play</span>
                        <span className={cn(
                            "text-sm font-mono font-bold",
                            epaPerPlay >= 0 ? "text-[var(--neon-green)]" : "text-[var(--destructive)]"
                        )}>
                            {epaPerPlay >= 0 ? "+" : ""}{epaPerPlay.toFixed(2)}
                        </span>
                    </div>
                    <div className="flex items-baseline gap-1">
                        <span className="text-[9px] text-[var(--muted-foreground)] font-medium">Comp%</span>
                        <span className="text-sm font-mono text-[var(--foreground)]/70 font-semibold">{completionPct}%</span>
                    </div>
                </div>
            </div>
        </motion.div>
    );
}

export default QuarterbackPerformanceCard;
