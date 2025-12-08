"use client";

import { ReactNode } from "react";
import { cn } from "@/lib/utils";
import { Clock, RefreshCw, Radio } from "lucide-react";
import { motion } from "framer-motion";

interface LiveLayoutShellProps {
    children: ReactNode;
    lastUpdated?: string;
    isRefreshing?: boolean;
}

export function LiveLayoutShell({ 
    children, 
    lastUpdated,
    isRefreshing = false 
}: LiveLayoutShellProps) {
    return (
        <div className="min-h-screen bg-[var(--background)] flex flex-col bg-grid-pattern">
            {/* Background gradient overlay */}
            <div className="fixed inset-0 bg-gradient-radial pointer-events-none" />
            
            {/* Page Header - Glass style */}
            <header className="sticky top-0 z-30 glass-panel">
                <div className="content-container page-padding">
                    <div className="flex items-center justify-between h-14 lg:h-16">
                        {/* Breadcrumb & Title */}
                        <div className="flex flex-col justify-center">
                            <nav className="flex items-center gap-1.5 text-xs text-[var(--muted-foreground)]">
                                <span className="hover:text-[var(--foreground)] transition-colors cursor-pointer">
                                    Predictr
                                </span>
                                <span className="opacity-50">/</span>
                                <span className="text-[var(--neon-blue)] font-medium">live</span>
                            </nav>
                            <div className="flex items-center gap-2">
                                <h1 className="text-lg lg:text-xl font-semibold text-[var(--foreground)] tracking-tight">
                                    Live Scoreboard
                                </h1>
                                <div className="neon-badge neon-badge-green text-[9px] flex items-center gap-1">
                                    <Radio size={10} />
                                    <span>LIVE</span>
                                </div>
                            </div>
                        </div>

                        {/* Last Updated Indicator */}
                        {lastUpdated && (
                            <motion.div 
                                className="flex items-center gap-2 px-3 py-1.5 rounded-lg glass-inner"
                                initial={{ opacity: 0, x: 20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ duration: 0.3 }}
                            >
                                <div className="relative">
                                    {isRefreshing ? (
                                        <RefreshCw size={14} className="animate-spin text-[var(--neon-blue)]" />
                                    ) : (
                                        <Clock size={14} className="text-[var(--muted-foreground)]" />
                                    )}
                                    {/* Live dot indicator */}
                                    <motion.div
                                        initial={{ scale: 0, opacity: 0 }}
                                        animate={{ scale: 1, opacity: 1 }}
                                        transition={{ duration: 0.2 }}
                                        className="absolute -top-0.5 -right-0.5 w-2 h-2 bg-[var(--neon-green)] rounded-full"
                                        key={lastUpdated}
                                    />
                                </div>
                                <div className="flex items-center gap-1.5 text-xs">
                                    <span className="text-[var(--muted-foreground)] hidden sm:inline">
                                        Updated:
                                    </span>
                                    <span className="font-mono font-medium text-[var(--foreground)]">
                                        {lastUpdated}
                                    </span>
                                </div>
                            </motion.div>
                        )}
                    </div>
                </div>
            </header>

            {/* Main Content */}
            <main className="flex-1 flex flex-col relative z-10">
                {children}
            </main>
        </div>
    );
}

export default LiveLayoutShell;
