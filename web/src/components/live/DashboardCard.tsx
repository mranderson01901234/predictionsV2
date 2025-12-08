"use client";

import { cn } from "@/lib/utils";
import { Skeleton } from "@/components/ui/skeleton";
import { ReactNode, forwardRef } from "react";

interface DashboardCardProps {
    children: ReactNode;
    className?: string;
    noPadding?: boolean;
    variant?: "default" | "elevated" | "hero";
    interactive?: boolean;
}

interface DashboardCardHeaderProps {
    title: string;
    subtitle?: string;
    action?: ReactNode;
    className?: string;
}

interface DashboardCardContentProps {
    children: ReactNode;
    className?: string;
}

// Main Card Component with glassmorphism
export const DashboardCard = forwardRef<HTMLDivElement, DashboardCardProps>(
    ({ children, className, noPadding = false, variant = "default", interactive = false }, ref) => {
        const variantClasses = {
            default: "glass-surface",
            elevated: "glass-surface-elevated",
            hero: "glass-surface-hero"
        };
        
        return (
            <div
                ref={ref}
                className={cn(
                    variantClasses[variant],
                    "transition-all duration-200",
                    interactive && "card-interactive cursor-pointer hover-glow-blue",
                    !noPadding && "p-4",
                    className
                )}
            >
                {children}
            </div>
        );
    }
);

DashboardCard.displayName = "DashboardCard";

// Card Header with consistent styling
export function DashboardCardHeader({
    title,
    subtitle,
    action,
    className,
}: DashboardCardHeaderProps) {
    return (
        <div className={cn("flex items-start justify-between mb-4", className)}>
            <div className="flex-1 min-w-0">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--muted-foreground)]">
                    {title}
                </h3>
                {subtitle && (
                    <p className="text-[10px] text-[var(--muted-foreground)] opacity-70 mt-0.5">
                        {subtitle}
                    </p>
                )}
            </div>
            {action && (
                <div className="flex-shrink-0 ml-4">
                    {action}
                </div>
            )}
        </div>
    );
}

// Card Content
export function DashboardCardContent({ children, className }: DashboardCardContentProps) {
    return (
        <div className={cn("flex-1 min-h-0", className)}>
            {children}
        </div>
    );
}

// Team Logo Header (for stats cards)
interface TeamLogoHeaderProps {
    awayLogo?: string;
    homeLogo?: string;
    awayTeam?: string;
    homeTeam?: string;
    className?: string;
}

export function TeamLogoHeader({
    awayLogo,
    homeLogo,
    awayTeam = "A",
    homeTeam = "H",
    className,
}: TeamLogoHeaderProps) {
    return (
        <div className={cn("flex items-center gap-3", className)}>
            {awayLogo ? (
                <img 
                    src={awayLogo} 
                    alt={awayTeam} 
                    className="w-5 h-5 object-contain opacity-70" 
                />
            ) : (
                <div className="w-5 h-5 rounded-full bg-[var(--glass-bg-elevated)] border border-[var(--glass-border)] flex items-center justify-center text-[9px] font-medium text-[var(--muted-foreground)]">
                    {awayTeam[0]}
                </div>
            )}
            {homeLogo ? (
                <img 
                    src={homeLogo} 
                    alt={homeTeam} 
                    className="w-5 h-5 object-contain" 
                />
            ) : (
                <div className="w-5 h-5 rounded-full bg-[var(--neon-blue-muted)] border border-[var(--neon-blue)]/20 flex items-center justify-center text-[9px] font-bold text-[var(--neon-blue)]">
                    {homeTeam[0]}
                </div>
            )}
        </div>
    );
}

// Skeleton Cards for Loading States - Glass style
export function DashboardCardSkeleton({ className }: { className?: string }) {
    return (
        <div className={cn(
            "glass-surface p-4 skeleton-shimmer",
            className
        )}>
            <Skeleton className="h-4 w-32 mb-4 bg-[var(--glass-bg-elevated)]" />
            <div className="space-y-3">
                <Skeleton className="h-8 w-full bg-[var(--glass-bg-elevated)]" />
                <Skeleton className="h-8 w-3/4 bg-[var(--glass-bg-elevated)]" />
                <Skeleton className="h-8 w-1/2 bg-[var(--glass-bg-elevated)]" />
            </div>
        </div>
    );
}

export function ChartSkeleton({ className }: { className?: string }) {
    return (
        <div className={cn(
            "glass-surface-elevated p-4 skeleton-shimmer",
            className
        )}>
            <Skeleton className="h-4 w-32 mb-4 bg-[var(--glass-bg-elevated)]" />
            <Skeleton className="h-48 w-full rounded-lg bg-[var(--glass-bg-elevated)]" />
        </div>
    );
}

export function QBCardSkeleton({ className }: { className?: string }) {
    return (
        <div className={cn(
            "glass-surface p-4 skeleton-shimmer",
            className
        )}>
            <Skeleton className="h-4 w-40 mb-4 bg-[var(--glass-bg-elevated)]" />
            <div className="grid grid-cols-2 gap-4">
                {[0, 1].map((i) => (
                    <div key={i} className="glass-inner p-3 space-y-3">
                        <div className="flex gap-3">
                            <Skeleton className="h-14 w-14 rounded-xl bg-[var(--glass-bg-elevated)]" />
                            <div className="flex-1 space-y-2">
                                <Skeleton className="h-4 w-24 bg-[var(--glass-bg-elevated)]" />
                                <Skeleton className="h-3 w-16 bg-[var(--glass-bg-elevated)]" />
                            </div>
                        </div>
                        <div className="grid grid-cols-4 gap-2">
                            <Skeleton className="h-12 bg-[var(--glass-bg-elevated)]" />
                            <Skeleton className="h-12 bg-[var(--glass-bg-elevated)]" />
                            <Skeleton className="h-12 bg-[var(--glass-bg-elevated)]" />
                            <Skeleton className="h-12 bg-[var(--glass-bg-elevated)]" />
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}

export function GameTileSkeleton() {
    return (
        <div className="min-w-[150px] p-3 rounded-xl glass-surface skeleton-shimmer">
            <Skeleton className="h-3 w-12 mb-3 bg-[var(--glass-bg-elevated)]" />
            <div className="space-y-2">
                <div className="flex justify-between items-center">
                    <div className="flex items-center gap-2">
                        <Skeleton className="h-5 w-5 rounded-full bg-[var(--glass-bg-elevated)]" />
                        <Skeleton className="h-4 w-16 bg-[var(--glass-bg-elevated)]" />
                    </div>
                    <Skeleton className="h-4 w-6 bg-[var(--glass-bg-elevated)]" />
                </div>
                <div className="flex justify-between items-center">
                    <div className="flex items-center gap-2">
                        <Skeleton className="h-5 w-5 rounded-full bg-[var(--glass-bg-elevated)]" />
                        <Skeleton className="h-4 w-16 bg-[var(--glass-bg-elevated)]" />
                    </div>
                    <Skeleton className="h-4 w-6 bg-[var(--glass-bg-elevated)]" />
                </div>
            </div>
        </div>
    );
}
