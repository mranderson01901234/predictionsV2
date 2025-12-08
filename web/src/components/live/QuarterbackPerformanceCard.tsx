"use client";

import { Quarterback } from "@/lib/mock_data";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
import { User, Zap } from "lucide-react";

interface QuarterbackPerformanceCardProps {
    homeQB: Quarterback;
    awayQB: Quarterback;
}

interface QBCapsuleProps {
    qb: Quarterback;
    isHome?: boolean;
    delay?: number;
}

// EPA Performance Meter - vertical bar style
function EPAMeter({ value, label, max = 15 }: { value: number; label: string; max?: number }) {
    const percentage = Math.min(Math.abs(value) / max * 100, 100);
    const isPositive = value >= 0;
    
    return (
        <div className="flex flex-col items-center gap-1.5">
            {/* Value label */}
            <span className={cn(
                "text-sm font-mono font-bold",
                isPositive ? "text-[var(--neon-green)]" : "text-[var(--destructive)]"
            )}>
                {value >= 0 ? "+" : ""}{value.toFixed(1)}
            </span>
            
            {/* Vertical bar container */}
            <div className="w-3 h-16 bg-[var(--glass-bg-elevated)] rounded-full overflow-hidden relative">
                <motion.div 
                    className={cn(
                        "absolute bottom-0 left-0 right-0 rounded-full transition-all duration-700",
                        isPositive 
                            ? "bg-gradient-to-t from-[var(--neon-green)] to-[var(--neon-green)]/60" 
                            : "bg-gradient-to-t from-[var(--destructive)] to-[var(--destructive)]/60"
                    )}
                    initial={{ height: 0 }}
                    animate={{ height: `${percentage}%` }}
                    transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1], delay: 0.3 }}
                    style={{
                        boxShadow: isPositive 
                            ? '0 0 8px var(--neon-green-glow)' 
                            : '0 0 8px rgba(255, 59, 92, 0.4)'
                    }}
                />
            </div>
            
            {/* Label */}
            <span className="text-[9px] text-[var(--muted-foreground)] uppercase tracking-wide font-medium">
                {label}
            </span>
        </div>
    );
}

// Individual QB Capsule with team color accent
function QBCapsule({ qb, isHome = false, delay = 0 }: QBCapsuleProps) {
    const epaPerPlay = qb.epa / (qb.attempts || 1);
    const completionPct = qb.attempts > 0 
        ? Math.round((qb.completions / qb.attempts) * 100) 
        : 0;

    return (
        <motion.div
            className="glass-surface-elevated relative overflow-hidden"
            initial={{ opacity: 0, y: 16, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{ duration: 0.5, delay, ease: [0.16, 1, 0.3, 1] }}
        >
            {/* Team color accent strip at top */}
            <div className={cn(
                "absolute top-0 left-0 right-0 h-1",
                isHome 
                    ? "bg-gradient-to-r from-[var(--neon-blue)] via-[var(--neon-blue)] to-transparent" 
                    : "bg-gradient-to-r from-transparent via-[var(--neon-purple)] to-[var(--neon-purple)]"
            )} />
            
            <div className="p-4">
                {/* Header: Photo + Name + Team Badge */}
                <div className="flex items-start gap-3 mb-4">
                    {/* Headshot with glow */}
                    <div className="relative flex-shrink-0">
                        <div className={cn(
                            "w-14 h-14 lg:w-16 lg:h-16 rounded-xl overflow-hidden border-2",
                            isHome 
                                ? "border-[var(--neon-blue)]/30" 
                                : "border-[var(--neon-purple)]/30"
                        )}>
                            <img
                                src={qb.headshot_url || ""}
                                alt={qb.name}
                                className="w-full h-full object-cover bg-[var(--muted)]"
                                onError={(e) => {
                                    const target = e.target as HTMLImageElement;
                                    target.style.display = 'none';
                                    const fallback = target.nextElementSibling as HTMLElement;
                                    if (fallback) fallback.style.display = 'flex';
                                }}
                            />
                            <div className="w-full h-full bg-[var(--glass-bg-elevated)] flex items-center justify-center hidden">
                                <User size={24} className="text-[var(--muted-foreground)]" />
                            </div>
                        </div>
                        
                        {/* Position badge */}
                        <div className={cn(
                            "absolute -bottom-1 -right-1 text-[8px] font-bold px-1.5 py-0.5 rounded-md",
                            isHome 
                                ? "bg-[var(--neon-blue-muted)] text-[var(--neon-blue)] border border-[var(--neon-blue)]/20" 
                                : "bg-[var(--neon-purple-muted)] text-[var(--neon-purple)] border border-[var(--neon-purple)]/20"
                        )}>
                            QB
                        </div>
                    </div>

                    {/* Name + Team */}
                    <div className="flex-1 min-w-0">
                        <div className="text-sm lg:text-base font-bold text-[var(--foreground)] truncate">
                            {qb.name}
                        </div>
                        <div className="text-[11px] text-[var(--muted-foreground)] font-medium mt-0.5">
                            {qb.team}
                        </div>
                        {/* QBR Mini badge */}
                        <div className="flex items-center gap-1.5 mt-1.5">
                            <span className="text-[9px] uppercase tracking-wide text-[var(--muted-foreground)]">QBR</span>
                            <span className={cn(
                                "text-xs font-mono font-semibold",
                                qb.qbr >= 70 ? "text-[var(--neon-green)]" : 
                                qb.qbr >= 50 ? "text-[var(--foreground)]" : "text-[var(--destructive)]"
                            )}>
                                {qb.qbr.toFixed(1)}
                            </span>
                        </div>
                    </div>
                </div>

                {/* Primary Stats Row */}
                <div className="grid grid-cols-4 gap-2 mb-4">
                    <div className="glass-inner p-2 text-center">
                        <div className="text-[9px] text-[var(--muted-foreground)] uppercase font-medium tracking-wide">
                            YDS
                        </div>
                        <div className="text-base font-bold font-mono text-[var(--foreground)] mt-0.5">
                            {qb.yards}
                        </div>
                    </div>
                    <div className="glass-inner p-2 text-center">
                        <div className="text-[9px] text-[var(--muted-foreground)] uppercase font-medium tracking-wide">
                            C/A
                        </div>
                        <div className="text-base font-bold font-mono text-[var(--foreground)] mt-0.5">
                            {qb.completions}/{qb.attempts}
                        </div>
                    </div>
                    <div className="glass-inner p-2 text-center">
                        <div className="text-[9px] text-[var(--muted-foreground)] uppercase font-medium tracking-wide">
                            TD
                        </div>
                        <div className="text-base font-bold font-mono text-[var(--neon-green)] mt-0.5">
                            {qb.tds}
                        </div>
                    </div>
                    <div className="glass-inner p-2 text-center">
                        <div className="text-[9px] text-[var(--muted-foreground)] uppercase font-medium tracking-wide">
                            INT
                        </div>
                        <div className={cn(
                            "text-base font-bold font-mono mt-0.5",
                            qb.ints > 0 ? "text-[var(--destructive)]" : "text-[var(--foreground)]"
                        )}>
                            {qb.ints}
                        </div>
                    </div>
                </div>

                {/* EPA Section with Meters */}
                <div className="glass-inner p-3">
                    <div className="flex items-center gap-2 mb-3">
                        <Zap size={12} className={cn(
                            isHome ? "text-[var(--neon-blue)]" : "text-[var(--neon-purple)]"
                        )} />
                        <span className="text-[10px] uppercase tracking-wider text-[var(--muted-foreground)] font-semibold">
                            EPA Performance
                        </span>
                    </div>
                    
                    <div className="flex items-end justify-around">
                        <EPAMeter value={qb.epa} label="Total" max={15} />
                        <EPAMeter value={epaPerPlay} label="Per Play" max={0.5} />
                    </div>
                </div>

                {/* Completion Percentage Bar */}
                <div className="mt-3">
                    <div className="flex items-center justify-between mb-1.5">
                        <span className="text-[10px] text-[var(--muted-foreground)] uppercase tracking-wide font-medium">
                            Completion %
                        </span>
                        <span className="text-xs font-mono font-semibold text-[var(--foreground)]">
                            {completionPct}%
                        </span>
                    </div>
                    <div className="h-1.5 bg-[var(--glass-bg-elevated)] rounded-full overflow-hidden">
                        <motion.div 
                            className={cn(
                                "h-full rounded-full",
                                isHome 
                                    ? "bg-gradient-to-r from-[var(--neon-blue)]/80 to-[var(--neon-blue)]" 
                                    : "bg-gradient-to-r from-[var(--neon-purple)]/80 to-[var(--neon-purple)]"
                            )}
                            initial={{ width: 0 }}
                            animate={{ width: `${completionPct}%` }}
                            transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1], delay: 0.4 }}
                        />
                    </div>
                </div>
            </div>
        </motion.div>
    );
}

export function QuarterbackPerformanceCard({ homeQB, awayQB }: QuarterbackPerformanceCardProps) {
    return (
        <motion.div
            className="h-full flex flex-col"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.3 }}
        >
            {/* Section Header */}
            <div className="flex items-center gap-2 mb-4">
                <div className="p-2 rounded-lg bg-[var(--glass-bg-elevated)] border border-[var(--glass-border)]">
                    <User size={14} className="text-[var(--neon-purple)]" />
                </div>
                <div>
                    <h3 className="text-sm font-semibold text-[var(--foreground)]">
                        Quarterback Performance
                    </h3>
                    <p className="text-[10px] text-[var(--muted-foreground)]">
                        Live EPA metrics & efficiency
                    </p>
                </div>
            </div>

            {/* QB Capsules Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 flex-1">
                <QBCapsule qb={awayQB} isHome={false} delay={0.1} />
                <QBCapsule qb={homeQB} isHome={true} delay={0.15} />
            </div>
        </motion.div>
    );
}

export default QuarterbackPerformanceCard;
