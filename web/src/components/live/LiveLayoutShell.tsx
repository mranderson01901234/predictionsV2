"use client";

import { ReactNode } from "react";
import { cn } from "@/lib/utils";
import { Clock, RefreshCw, Radio, Activity, LayoutDashboard, Users, Trophy, Calculator, ChevronDown } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

interface LiveLayoutShellProps {
    children: ReactNode;
    lastUpdated?: string;
    isRefreshing?: boolean;
}

// Navigation items for header
const NAV_ITEMS = [
    { label: "Live", href: "/games", icon: Activity, isLive: true },
    {
        label: "Platform",
        icon: LayoutDashboard,
        children: [
            { label: "Predictions", href: "/predictions" },
            { label: "Confidence Pool", href: "/confidence" },
            { label: "Betting Card", href: "/betting" },
            { label: "Model Performance", href: "/performance" },
        ],
    },
    {
        label: "Teams",
        icon: Users,
        children: [
            { label: "Power Ratings", href: "/power-ratings" },
            { label: "QBs", href: "/qb-rankings" },
            { label: "Receivers", href: "/receivers" },
        ],
    },
    {
        label: "Tools",
        icon: Calculator,
        children: [
            { label: "Calculators", href: "/calculators" },
        ],
    },
];

function NavDropdown({ item }: { item: typeof NAV_ITEMS[0] }) {
    const [open, setOpen] = useState(false);
    const pathname = usePathname();
    const Icon = item.icon;

    if (!item.children) {
        return (
            <Link
                href={item.href || "#"}
                className={cn(
                    "flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg transition-colors",
                    pathname === item.href
                        ? "bg-white/10 text-white"
                        : "text-white/60 hover:text-white hover:bg-white/5"
                )}
            >
                <Icon size={14} />
                <span>{item.label}</span>
                {item.isLive && (
                    <span className="relative flex h-1.5 w-1.5 ml-1">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                        <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-emerald-400" />
                    </span>
                )}
            </Link>
        );
    }

    return (
        <div
            className="relative"
            onMouseEnter={() => setOpen(true)}
            onMouseLeave={() => setOpen(false)}
        >
            <button
                className={cn(
                    "flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg transition-colors",
                    open ? "bg-white/10 text-white" : "text-white/60 hover:text-white hover:bg-white/5"
                )}
            >
                <Icon size={14} />
                <span>{item.label}</span>
                <ChevronDown size={12} className={cn("transition-transform", open && "rotate-180")} />
            </button>
            <AnimatePresence>
                {open && (
                    <motion.div
                        initial={{ opacity: 0, y: 4 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: 4 }}
                        transition={{ duration: 0.15 }}
                        className="absolute top-full left-0 mt-1 min-w-[160px] bg-[#151515] border border-white/10 rounded-lg shadow-xl py-1 z-50"
                    >
                        {item.children.map((child) => (
                            <Link
                                key={child.href}
                                href={child.href}
                                className={cn(
                                    "block px-3 py-2 text-xs transition-colors",
                                    pathname === child.href
                                        ? "text-white bg-white/5"
                                        : "text-white/60 hover:text-white hover:bg-white/5"
                                )}
                            >
                                {child.label}
                            </Link>
                        ))}
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}

export function LiveLayoutShell({
    children,
    lastUpdated,
    isRefreshing = false,
}: LiveLayoutShellProps) {
    return (
        <div className="h-screen bg-[#0a0a0a] flex flex-col overflow-hidden">
            {/* Compact Header */}
            <header className="flex-shrink-0 z-40 h-12 bg-[#0d0d0d] border-b border-white/5 flex items-center px-4">
                {/* Left: Title */}
                <div className="flex items-center gap-3">
                    <div className="flex flex-col">
                        <nav className="flex items-center gap-1 text-[10px] text-white/40">
                            <span>Predictr</span>
                            <span>/</span>
                            <span className="text-emerald-400">live</span>
                        </nav>
                        <div className="flex items-center gap-2">
                            <h1 className="text-sm font-semibold text-white">Live Scoreboard</h1>
                            <div className="flex items-center gap-1 px-1.5 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/20">
                                <Radio size={10} className="text-emerald-400" />
                                <span className="text-[9px] font-medium text-emerald-400">LIVE</span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Center: Navigation Tabs */}
                <nav className="flex items-center gap-1 ml-8">
                    {NAV_ITEMS.map((item) => (
                        <NavDropdown key={item.label} item={item} />
                    ))}
                </nav>

                {/* Right: Last Updated */}
                <div className="ml-auto">
                    {lastUpdated && (
                        <div className="flex items-center gap-2 px-2.5 py-1 rounded-lg bg-white/[0.02] border border-white/5">
                            <div className="relative">
                                {isRefreshing ? (
                                    <RefreshCw size={12} className="animate-spin text-emerald-400" />
                                ) : (
                                    <Clock size={12} className="text-white/40" />
                                )}
                            </div>
                            <span className="text-[10px] text-white/40">Updated:</span>
                            <span className="text-[10px] font-mono text-white/70">{lastUpdated}</span>
                        </div>
                    )}
                </div>
            </header>

            {/* Main Content */}
            <main className="flex-1 flex relative overflow-hidden">
                {children}
            </main>
        </div>
    );
}

export default LiveLayoutShell;
