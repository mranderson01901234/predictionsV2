"use client";

import { useState, createContext, useContext } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from "@/components/ui/tooltip";
import { Sheet, SheetContent, SheetTrigger, SheetTitle, SheetHeader } from "@/components/ui/sheet";
import {
    Activity,
    LayoutDashboard,
    Trophy,
    Users,
    Calculator,
    ChevronDown,
    ChevronRight,
    Menu,
    ChevronLeft,
    Target,
    TrendingUp,
    BarChart3,
    Radio,
    ExternalLink,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

// Sidebar context for collapse state
const SidebarContext = createContext<{
    collapsed: boolean;
    setCollapsed: (collapsed: boolean) => void;
    mobileOpen: boolean;
    setMobileOpen: (open: boolean) => void;
}>({
    collapsed: false,
    setCollapsed: () => {},
    mobileOpen: false,
    setMobileOpen: () => {},
});

export const useSidebar = () => useContext(SidebarContext);

interface NavItem {
    label: string;
    href?: string;
    icon?: React.ElementType;
    children?: NavItem[];
    isLive?: boolean;
    external?: boolean;
}

interface NavSection {
    label: string;
    items: NavItem[];
}

const NAV_SECTIONS: NavSection[] = [
    {
        label: "Live",
        items: [
            {
                label: "Live",
                href: "/games",
                icon: Activity,
                isLive: true
            }
        ]
    },
    {
        label: "Platform",
        items: [
            {
                label: "Platform",
                icon: LayoutDashboard,
                children: [
                    { label: "Predictions", href: "/predictions", icon: Target },
                    { label: "Confidence Pool", href: "/confidence", icon: BarChart3 },
                    { label: "Betting Card", href: "/betting", icon: TrendingUp },
                    { label: "Model Performance", href: "/performance", icon: Radio }
                ]
            }
        ]
    },
    {
        label: "Data",
        items: [
            {
                label: "Teams",
                icon: Users,
                children: [
                    { label: "Power Ratings", href: "/power-ratings" },
                    { label: "EPA Tiers", href: "/epa-tiers" },
                    { label: "Schedules", href: "/schedules" }
                ]
            },
            {
                label: "QBs",
                icon: Trophy,
                children: [
                    { label: "Rankings", href: "/qb-rankings" },
                    { label: "Era Adjustments", href: "/era-adj" }
                ]
            }
        ]
    },
    {
        label: "Tools",
        items: [
            {
                label: "Tools",
                icon: Calculator,
                children: [
                    { label: "Receivers", href: "/receivers" },
                    { label: "Calculators", href: "/calculators" }
                ]
            }
        ]
    }
];

// Navigation Item Component
function NavItem({ 
    item, 
    collapsed, 
    depth = 0 
}: { 
    item: NavItem; 
    collapsed: boolean;
    depth?: number;
}) {
    const pathname = usePathname();
    const [expanded, setExpanded] = useState(
        item.children?.some(child => child.href === pathname) || false
    );
    
    const isActive = item.href === pathname;
    const hasActiveChild = item.children?.some(child => child.href === pathname);
    const Icon = item.icon;

    // Single link item
    if (!item.children) {
        const linkContent = (
            <Link
                href={item.href || "#"}
                className={cn(
                    "group relative flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-all duration-150",
                    "hover:bg-[var(--sidebar-accent)]",
                    isActive 
                        ? "bg-[var(--accent-blue-muted)] text-[var(--accent-blue)]" 
                        : "text-[var(--sidebar-muted)] hover:text-[var(--sidebar-foreground)]",
                    depth > 0 && "ml-6 text-[13px]",
                    collapsed && depth === 0 && "justify-center px-2"
                )}
            >
                {/* Active indicator bar */}
                {isActive && depth === 0 && (
                    <motion.div
                        layoutId="activeNavIndicator"
                        className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-5 bg-[var(--accent-blue)] rounded-full"
                        transition={{ type: "spring", bounce: 0.2, duration: 0.4 }}
                    />
                )}
                
                {Icon && (
                    <Icon 
                        size={collapsed && depth === 0 ? 20 : 16} 
                        className={cn(
                            "shrink-0 transition-colors",
                            isActive ? "text-[var(--accent-blue)]" : "text-inherit"
                        )} 
                    />
                )}
                
                {(!collapsed || depth > 0) && (
                    <span className="truncate">{item.label}</span>
                )}
                
                {item.isLive && !collapsed && (
                    <span className="ml-auto relative flex h-2 w-2">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[var(--success)] opacity-75"></span>
                        <span className="relative inline-flex rounded-full h-2 w-2 bg-[var(--success)]"></span>
                    </span>
                )}
                
                {item.external && !collapsed && (
                    <ExternalLink size={12} className="ml-auto opacity-50" />
                )}
            </Link>
        );

        // Wrap in tooltip when collapsed
        if (collapsed && depth === 0) {
            return (
                <Tooltip>
                    <TooltipTrigger asChild>
                        {linkContent}
                    </TooltipTrigger>
                    <TooltipContent side="right" className="flex items-center gap-2">
                        {item.label}
                        {item.isLive && (
                            <span className="relative flex h-2 w-2">
                                <span className="relative inline-flex rounded-full h-2 w-2 bg-[var(--success)]"></span>
                            </span>
                        )}
                    </TooltipContent>
                </Tooltip>
            );
        }

        return linkContent;
    }

    // Expandable section
    const triggerContent = (
        <button
            onClick={() => setExpanded(!expanded)}
            className={cn(
                "group relative flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-all duration-150",
                "hover:bg-[var(--sidebar-accent)]",
                hasActiveChild 
                    ? "text-[var(--sidebar-foreground)]" 
                    : "text-[var(--sidebar-muted)] hover:text-[var(--sidebar-foreground)]",
                collapsed && "justify-center px-2"
            )}
        >
            {/* Active indicator for sections with active children */}
            {hasActiveChild && (
                <div className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-5 bg-[var(--accent-blue)] rounded-full" />
            )}
            
            {Icon && (
                <Icon 
                    size={collapsed ? 20 : 16} 
                    className="shrink-0" 
                />
            )}
            
            {!collapsed && (
                <>
                    <span className="truncate flex-1 text-left">{item.label}</span>
                    <ChevronDown 
                        size={14} 
                        className={cn(
                            "shrink-0 opacity-50 transition-transform duration-200",
                            expanded && "rotate-180"
                        )} 
                    />
                </>
            )}
        </button>
    );

    // Wrap in tooltip when collapsed
    if (collapsed) {
        return (
            <Tooltip>
                <TooltipTrigger asChild>
                    {triggerContent}
                </TooltipTrigger>
                <TooltipContent side="right" className="p-0">
                    <div className="py-1">
                        <div className="px-3 py-1.5 text-xs font-medium text-muted-foreground">
                            {item.label}
                        </div>
                        {item.children?.map((child) => (
                            <Link
                                key={child.href}
                                href={child.href || "#"}
                                className={cn(
                                    "block px-3 py-1.5 text-sm transition-colors",
                                    pathname === child.href
                                        ? "text-[var(--accent-blue)]"
                                        : "hover:text-foreground text-muted-foreground"
                                )}
                            >
                                {child.label}
                            </Link>
                        ))}
                    </div>
                </TooltipContent>
            </Tooltip>
        );
    }

    return (
        <div>
            {triggerContent}
            <AnimatePresence initial={false}>
                {expanded && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: "auto", opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.2, ease: "easeInOut" }}
                        className="overflow-hidden"
                    >
                        <div className="pt-1 space-y-0.5">
                            {item.children.map((child) => (
                                <NavItem 
                                    key={child.href} 
                                    item={child} 
                                    collapsed={false}
                                    depth={depth + 1}
                                />
                            ))}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}

// Desktop Sidebar
function DesktopSidebar({ collapsed, onCollapse }: { collapsed: boolean; onCollapse: (val: boolean) => void }) {
    return (
        <aside 
            className={cn(
                "fixed top-0 left-0 h-screen bg-[var(--sidebar)] border-r border-[var(--sidebar-border)] z-40 hidden lg:flex flex-col transition-all duration-300 ease-in-out",
                collapsed ? "w-16" : "w-64"
            )}
        >
            {/* Logo Section */}
            <div className={cn(
                "h-16 flex items-center border-b border-[var(--sidebar-border)] shrink-0",
                collapsed ? "justify-center px-2" : "px-4"
            )}>
                <Link href="/" className="flex items-center gap-3">
                    <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center shrink-0">
                        <span className="font-bold text-white text-sm">P</span>
                    </div>
                    {!collapsed && (
                        <motion.span 
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            className="font-bold text-lg text-[var(--sidebar-foreground)] tracking-tight"
                        >
                            Predictr
                        </motion.span>
                    )}
                </Link>
            </div>

            {/* Navigation */}
            <ScrollArea className="flex-1 py-4">
                <TooltipProvider delayDuration={0}>
                    <nav className={cn("space-y-6", collapsed ? "px-2" : "px-3")}>
                        {NAV_SECTIONS.map((section, index) => (
                            <div key={section.label}>
                                {/* Section Label */}
                                {!collapsed && index > 0 && (
                                    <div className="px-3 mb-2">
                                        <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--sidebar-muted)] opacity-60">
                                            {section.label}
                                        </span>
                                    </div>
                                )}
                                {collapsed && index > 0 && (
                                    <Separator className="my-2" />
                                )}
                                
                                {/* Section Items */}
                                <div className="space-y-0.5">
                                    {section.items.map((item) => (
                                        <NavItem 
                                            key={item.label} 
                                            item={item} 
                                            collapsed={collapsed} 
                                        />
                                    ))}
                                </div>
                            </div>
                        ))}
                    </nav>
                </TooltipProvider>
            </ScrollArea>

            {/* Collapse Toggle */}
            <div className={cn(
                "p-3 border-t border-[var(--sidebar-border)]",
                collapsed && "flex justify-center"
            )}>
                <Button
                    variant="ghost"
                    size={collapsed ? "icon" : "sm"}
                    onClick={() => onCollapse(!collapsed)}
                    className={cn(
                        "text-[var(--sidebar-muted)] hover:text-[var(--sidebar-foreground)]",
                        !collapsed && "w-full justify-start gap-2"
                    )}
                >
                    <ChevronLeft 
                        size={16} 
                        className={cn(
                            "transition-transform duration-300",
                            collapsed && "rotate-180"
                        )} 
                    />
                    {!collapsed && <span className="text-sm">Collapse</span>}
                </Button>
            </div>

            {/* Affiliate Section - only when expanded */}
            {!collapsed && (
                <div className="p-3 border-t border-[var(--sidebar-border)]">
                    <div className="bg-gradient-to-br from-[var(--card)] to-[var(--secondary)] rounded-xl p-4 border border-[var(--border)]">
                        <div className="flex items-center gap-2 mb-2">
                            <span className="text-xs font-bold text-[var(--foreground)]">Kalshi $10 Bonus</span>
                            <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-[var(--success-muted)] text-[var(--success)] font-medium">
                                $10
                            </span>
                        </div>
                        <p className="text-[11px] text-[var(--muted-foreground)] mb-3 leading-relaxed">
                            Sign up and get $10 free to trade on NFL games.
                        </p>
                        <Button 
                            size="sm" 
                            className="w-full bg-[var(--accent-blue)] hover:bg-[var(--accent-blue)]/90 text-white text-xs font-semibold"
                        >
                            Claim Now
                            <ExternalLink size={12} className="ml-1.5" />
                        </Button>
                    </div>
                </div>
            )}
        </aside>
    );
}

// Mobile Sidebar (Sheet)
function MobileSidebar({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
    return (
        <Sheet open={open} onOpenChange={onOpenChange}>
            <SheetContent side="left" className="w-72 p-0 bg-[var(--sidebar)] border-[var(--sidebar-border)]">
                <SheetHeader className="h-16 flex-row items-center border-b border-[var(--sidebar-border)] px-4 space-y-0">
                    <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center shrink-0">
                        <span className="font-bold text-white text-sm">P</span>
                    </div>
                    <SheetTitle className="ml-3 font-bold text-lg tracking-tight">Predictr</SheetTitle>
                </SheetHeader>
                
                <ScrollArea className="flex-1 py-4">
                    <TooltipProvider delayDuration={0}>
                        <nav className="space-y-6 px-3">
                            {NAV_SECTIONS.map((section, index) => (
                                <div key={section.label}>
                                    {index > 0 && (
                                        <div className="px-3 mb-2">
                                            <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--sidebar-muted)] opacity-60">
                                                {section.label}
                                            </span>
                                        </div>
                                    )}
                                    <div className="space-y-0.5">
                                        {section.items.map((item) => (
                                            <NavItem 
                                                key={item.label} 
                                                item={item} 
                                                collapsed={false} 
                                            />
                                        ))}
                                    </div>
                                </div>
                            ))}
                        </nav>
                    </TooltipProvider>
                </ScrollArea>

                {/* Affiliate Section */}
                <div className="p-3 border-t border-[var(--sidebar-border)] mt-auto">
                    <div className="bg-gradient-to-br from-[var(--card)] to-[var(--secondary)] rounded-xl p-4 border border-[var(--border)]">
                        <div className="flex items-center gap-2 mb-2">
                            <span className="text-xs font-bold text-[var(--foreground)]">Kalshi $10 Bonus</span>
                            <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-[var(--success-muted)] text-[var(--success)] font-medium">
                                $10
                            </span>
                        </div>
                        <p className="text-[11px] text-[var(--muted-foreground)] mb-3 leading-relaxed">
                            Sign up and get $10 free to trade on NFL games.
                        </p>
                        <Button 
                            size="sm" 
                            className="w-full bg-[var(--accent-blue)] hover:bg-[var(--accent-blue)]/90 text-white text-xs font-semibold"
                        >
                            Claim Now
                            <ExternalLink size={12} className="ml-1.5" />
                        </Button>
                    </div>
                </div>
            </SheetContent>
        </Sheet>
    );
}

// Mobile Menu Button
export function MobileMenuButton() {
    const { setMobileOpen } = useSidebar();
    
    return (
        <Button
            variant="ghost"
            size="icon"
            onClick={() => setMobileOpen(true)}
            className="lg:hidden fixed top-4 left-4 z-50 bg-[var(--card)] border border-[var(--border)] hover:bg-[var(--secondary)]"
        >
            <Menu size={20} />
        </Button>
    );
}

// Main Sidebar Component
export function Sidebar() {
    const [collapsed, setCollapsed] = useState(false);
    const [mobileOpen, setMobileOpen] = useState(false);

    return (
        <SidebarContext.Provider value={{ collapsed, setCollapsed, mobileOpen, setMobileOpen }}>
            {/* Desktop */}
            <DesktopSidebar collapsed={collapsed} onCollapse={setCollapsed} />
            
            {/* Mobile */}
            <MobileSidebar open={mobileOpen} onOpenChange={setMobileOpen} />
            <MobileMenuButton />
        </SidebarContext.Provider>
    );
}

// Export context hook for layout adjustments
export { SidebarContext };
