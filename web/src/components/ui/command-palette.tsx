"use client";

import { useState, useEffect, useCallback, useMemo } from "react";
import { useRouter } from "next/navigation";
import * as Dialog from "@radix-ui/react-dialog";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import { 
    Search, 
    Activity, 
    BarChart3, 
    Users, 
    Trophy, 
    Calculator,
    ArrowRight,
    Command,
    TrendingUp,
    Target,
    Radio
} from "lucide-react";

interface CommandItem {
    id: string;
    label: string;
    description?: string;
    icon: React.ReactNode;
    href?: string;
    action?: () => void;
    category: string;
    keywords?: string[];
}

const COMMAND_ITEMS: CommandItem[] = [
    // Live
    {
        id: "live",
        label: "Live Scoreboard",
        description: "Real-time game data and AI predictions",
        icon: <Activity size={16} className="text-[var(--neon-green)]" />,
        href: "/games",
        category: "Navigation",
        keywords: ["games", "scores", "live", "scoreboard"]
    },
    // Platform
    {
        id: "predictions",
        label: "Predictions",
        description: "Model predictions and confidence scores",
        icon: <Target size={16} className="text-[var(--neon-blue)]" />,
        href: "/predictions",
        category: "Platform",
        keywords: ["predict", "model", "forecast"]
    },
    {
        id: "confidence",
        label: "Confidence Pool",
        description: "Weekly confidence pick rankings",
        icon: <BarChart3 size={16} className="text-[var(--neon-purple)]" />,
        href: "/confidence",
        category: "Platform",
        keywords: ["pool", "confidence", "picks"]
    },
    {
        id: "betting",
        label: "Betting Card",
        description: "Today's recommended plays",
        icon: <TrendingUp size={16} className="text-[var(--neon-green)]" />,
        href: "/betting",
        category: "Platform",
        keywords: ["bet", "plays", "card"]
    },
    {
        id: "performance",
        label: "Model Performance",
        description: "Historical accuracy metrics",
        icon: <Radio size={16} className="text-[var(--warning)]" />,
        href: "/performance",
        category: "Platform",
        keywords: ["accuracy", "history", "performance"]
    },
    // Data
    {
        id: "power-ratings",
        label: "Power Ratings",
        description: "Team strength rankings",
        icon: <Users size={16} className="text-[var(--neon-blue)]" />,
        href: "/power-ratings",
        category: "Data",
        keywords: ["power", "ratings", "teams", "rankings"]
    },
    {
        id: "qb-rankings",
        label: "QB Rankings",
        description: "Quarterback performance tiers",
        icon: <Trophy size={16} className="text-[var(--warning)]" />,
        href: "/qb-rankings",
        category: "Data",
        keywords: ["qb", "quarterback", "rankings"]
    },
    {
        id: "receivers",
        label: "Receiving Leaders",
        description: "Top receiving performers",
        icon: <Calculator size={16} className="text-[var(--neon-purple)]" />,
        href: "/receivers",
        category: "Tools",
        keywords: ["receivers", "receiving", "leaders"]
    }
];

interface CommandPaletteProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
}

export function CommandPalette({ open, onOpenChange }: CommandPaletteProps) {
    const [search, setSearch] = useState("");
    const [selectedIndex, setSelectedIndex] = useState(0);
    const router = useRouter();

    // Filter items based on search
    const filteredItems = useMemo(() => {
        if (!search.trim()) return COMMAND_ITEMS;
        
        const query = search.toLowerCase();
        return COMMAND_ITEMS.filter(item => {
            const matchLabel = item.label.toLowerCase().includes(query);
            const matchDescription = item.description?.toLowerCase().includes(query);
            const matchKeywords = item.keywords?.some(k => k.includes(query));
            return matchLabel || matchDescription || matchKeywords;
        });
    }, [search]);

    // Group items by category
    const groupedItems = useMemo(() => {
        const groups: Record<string, CommandItem[]> = {};
        filteredItems.forEach(item => {
            if (!groups[item.category]) {
                groups[item.category] = [];
            }
            groups[item.category].push(item);
        });
        return groups;
    }, [filteredItems]);

    // Reset selection when search changes
    useEffect(() => {
        setSelectedIndex(0);
    }, [search]);

    // Reset search when closed
    useEffect(() => {
        if (!open) {
            setSearch("");
            setSelectedIndex(0);
        }
    }, [open]);

    // Execute selected item
    const executeItem = useCallback((item: CommandItem) => {
        if (item.href) {
            router.push(item.href);
        } else if (item.action) {
            item.action();
        }
        onOpenChange(false);
    }, [router, onOpenChange]);

    // Keyboard navigation
    const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
        if (e.key === "ArrowDown") {
            e.preventDefault();
            setSelectedIndex(prev => Math.min(prev + 1, filteredItems.length - 1));
        } else if (e.key === "ArrowUp") {
            e.preventDefault();
            setSelectedIndex(prev => Math.max(prev - 1, 0));
        } else if (e.key === "Enter") {
            e.preventDefault();
            if (filteredItems[selectedIndex]) {
                executeItem(filteredItems[selectedIndex]);
            }
        }
    }, [filteredItems, selectedIndex, executeItem]);

    return (
        <Dialog.Root open={open} onOpenChange={onOpenChange}>
            <AnimatePresence>
                {open && (
                    <Dialog.Portal forceMount>
                        <Dialog.Overlay asChild>
                            <motion.div
                                className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm"
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0 }}
                                transition={{ duration: 0.15 }}
                            />
                        </Dialog.Overlay>
                        
                        <Dialog.Content asChild>
                            <motion.div
                                className={cn(
                                    "fixed left-1/2 top-[20%] z-50 w-full max-w-lg -translate-x-1/2",
                                    "glass-surface-hero overflow-hidden shadow-2xl"
                                )}
                                initial={{ opacity: 0, y: -20, scale: 0.95 }}
                                animate={{ opacity: 1, y: 0, scale: 1 }}
                                exit={{ opacity: 0, y: -20, scale: 0.95 }}
                                transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
                            >
                                {/* Search Input */}
                                <div className="flex items-center gap-3 px-4 py-3 border-b border-[var(--glass-border)]">
                                    <Search size={18} className="text-[var(--muted-foreground)]" />
                                    <input
                                        type="text"
                                        placeholder="Search commands..."
                                        value={search}
                                        onChange={(e) => setSearch(e.target.value)}
                                        onKeyDown={handleKeyDown}
                                        className="flex-1 bg-transparent text-sm text-[var(--foreground)] placeholder:text-[var(--muted-foreground)] outline-none"
                                        autoFocus
                                    />
                                    <kbd className="px-2 py-0.5 text-[10px] font-mono text-[var(--muted-foreground)] bg-[var(--glass-bg-elevated)] border border-[var(--glass-border)] rounded">
                                        ESC
                                    </kbd>
                                </div>

                                {/* Results */}
                                <div className="max-h-[320px] overflow-y-auto scrollbar-thin py-2">
                                    {filteredItems.length === 0 ? (
                                        <div className="px-4 py-8 text-center">
                                            <p className="text-sm text-[var(--muted-foreground)]">
                                                No results found for "{search}"
                                            </p>
                                        </div>
                                    ) : (
                                        Object.entries(groupedItems).map(([category, items]) => (
                                            <div key={category} className="mb-2 last:mb-0">
                                                <div className="px-4 py-1.5">
                                                    <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--muted-foreground)]">
                                                        {category}
                                                    </span>
                                                </div>
                                                {items.map((item) => {
                                                    const globalIndex = filteredItems.indexOf(item);
                                                    const isSelected = globalIndex === selectedIndex;
                                                    
                                                    return (
                                                        <button
                                                            key={item.id}
                                                            onClick={() => executeItem(item)}
                                                            onMouseEnter={() => setSelectedIndex(globalIndex)}
                                                            className={cn(
                                                                "w-full flex items-center gap-3 px-4 py-2.5 text-left transition-colors",
                                                                isSelected 
                                                                    ? "bg-[var(--glass-bg-elevated)]" 
                                                                    : "hover:bg-[var(--glass-bg-subtle)]"
                                                            )}
                                                        >
                                                            <div className={cn(
                                                                "p-1.5 rounded-lg",
                                                                isSelected 
                                                                    ? "bg-[var(--neon-blue-muted)] border border-[var(--neon-blue)]/20" 
                                                                    : "bg-[var(--glass-bg)] border border-[var(--glass-border)]"
                                                            )}>
                                                                {item.icon}
                                                            </div>
                                                            <div className="flex-1 min-w-0">
                                                                <div className={cn(
                                                                    "text-sm font-medium truncate",
                                                                    isSelected 
                                                                        ? "text-[var(--foreground)]" 
                                                                        : "text-[var(--muted-foreground)]"
                                                                )}>
                                                                    {item.label}
                                                                </div>
                                                                {item.description && (
                                                                    <div className="text-[11px] text-[var(--muted-foreground)] truncate opacity-70">
                                                                        {item.description}
                                                                    </div>
                                                                )}
                                                            </div>
                                                            {isSelected && (
                                                                <ArrowRight size={14} className="text-[var(--neon-blue)]" />
                                                            )}
                                                        </button>
                                                    );
                                                })}
                                            </div>
                                        ))
                                    )}
                                </div>

                                {/* Footer */}
                                <div className="flex items-center justify-between px-4 py-2 border-t border-[var(--glass-border)] bg-[var(--glass-bg)]">
                                    <div className="flex items-center gap-3">
                                        <div className="flex items-center gap-1">
                                            <kbd className="px-1.5 py-0.5 text-[9px] font-mono text-[var(--muted-foreground)] bg-[var(--glass-bg-elevated)] border border-[var(--glass-border)] rounded">
                                                ↑
                                            </kbd>
                                            <kbd className="px-1.5 py-0.5 text-[9px] font-mono text-[var(--muted-foreground)] bg-[var(--glass-bg-elevated)] border border-[var(--glass-border)] rounded">
                                                ↓
                                            </kbd>
                                            <span className="text-[10px] text-[var(--muted-foreground)] ml-1">Navigate</span>
                                        </div>
                                        <div className="flex items-center gap-1">
                                            <kbd className="px-1.5 py-0.5 text-[9px] font-mono text-[var(--muted-foreground)] bg-[var(--glass-bg-elevated)] border border-[var(--glass-border)] rounded">
                                                ↵
                                            </kbd>
                                            <span className="text-[10px] text-[var(--muted-foreground)] ml-1">Select</span>
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-1.5">
                                        <Command size={12} className="text-[var(--neon-blue)]" />
                                        <span className="text-[10px] text-[var(--muted-foreground)]">
                                            Predictr Command Palette
                                        </span>
                                    </div>
                                </div>
                            </motion.div>
                        </Dialog.Content>
                    </Dialog.Portal>
                )}
            </AnimatePresence>
        </Dialog.Root>
    );
}

// Hook to manage command palette state globally
export function useCommandPalette() {
    const [open, setOpen] = useState(false);

    // Global keyboard shortcut
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if ((e.metaKey || e.ctrlKey) && e.key === "k") {
                e.preventDefault();
                setOpen(prev => !prev);
            }
        };

        window.addEventListener("keydown", handleKeyDown);
        return () => window.removeEventListener("keydown", handleKeyDown);
    }, []);

    return { open, setOpen };
}

export default CommandPalette;

