"use client";

import { forwardRef, ReactNode } from "react";
import { motion, HTMLMotionProps } from "framer-motion";
import { cn } from "@/lib/utils";
import { LucideIcon } from "lucide-react";

/* ============================================
   CARD VARIANTS (Bloomberg Terminal-Grade)
   ============================================ */

type CardVariant = "primary" | "secondary" | "tertiary" | "default" | "elevated" | "hero";

interface GlassCardProps extends Omit<HTMLMotionProps<"div">, "children"> {
    children: ReactNode;
    variant?: CardVariant;
    className?: string;
    noPadding?: boolean;
    interactive?: boolean;
}

const variantClasses: Record<CardVariant, string> = {
    primary: "card-primary",
    secondary: "card-secondary",
    tertiary: "card-tertiary",
    default: "glass-surface",
    elevated: "glass-surface-elevated",
    hero: "glass-surface-hero",
};

export const GlassCard = forwardRef<HTMLDivElement, GlassCardProps>(
    ({
        children,
        variant = "secondary",
        className,
        noPadding = false,
        interactive = false,
        ...props
    }, ref) => {
        return (
            <motion.div
                ref={ref}
                className={cn(
                    variantClasses[variant],
                    !noPadding && "p-4",
                    interactive && "cursor-pointer",
                    className
                )}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
                {...props}
            >
                {children}
            </motion.div>
        );
    }
);

GlassCard.displayName = "GlassCard";

/* ============================================
   GLASS CARD HEADER (Simplified)
   ============================================ */

interface GlassCardHeaderProps {
    title: string;
    subtitle?: string;
    icon?: LucideIcon;
    action?: ReactNode;
    badge?: {
        text: string;
        variant: "blue" | "green" | "purple" | "warning" | "destructive";
    };
    className?: string;
}

const badgeVariantClasses = {
    blue: "neon-badge-blue",
    green: "neon-badge-green",
    purple: "neon-badge-purple",
    warning: "neon-badge-warning",
    destructive: "neon-badge-destructive",
};

export function GlassCardHeader({
    title,
    subtitle,
    icon: Icon,
    action,
    badge,
    className,
}: GlassCardHeaderProps) {
    return (
        <div className={cn("flex items-start justify-between mb-4", className)}>
            <div className="flex items-start gap-3">
                {Icon && (
                    <div className="p-1.5 rounded-lg bg-white/5 border border-white/5">
                        <Icon size={14} className="text-white/50" />
                    </div>
                )}
                <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                        <h3 className="text-sm font-medium text-white/90 tracking-tight">
                            {title}
                        </h3>
                        {badge && (
                            <span className={cn("neon-badge", badgeVariantClasses[badge.variant])}>
                                {badge.text}
                            </span>
                        )}
                    </div>
                    {subtitle && (
                        <p className="text-[11px] text-white/40 mt-0.5">
                            {subtitle}
                        </p>
                    )}
                </div>
            </div>
            {action && (
                <div className="flex-shrink-0 ml-4">
                    {action}
                </div>
            )}
        </div>
    );
}

/* ============================================
   GLASS PANEL (Full-width sections)
   ============================================ */

interface GlassPanelProps {
    children: ReactNode;
    className?: string;
}

export function GlassPanel({ children, className }: GlassPanelProps) {
    return (
        <div className={cn(
            "w-full",
            "bg-white/[0.02]",
            "backdrop-blur-md",
            "border-b border-white/5",
            className
        )}>
            {children}
        </div>
    );
}

/* ============================================
   STAT VALUE DISPLAY (Clean, no glow)
   ============================================ */

interface StatValueProps {
    value: string | number;
    label?: string;
    size?: "sm" | "md" | "lg" | "xl";
    variant?: "positive" | "negative" | "neutral" | "default";
    className?: string;
}

const sizeClasses = {
    sm: "text-lg",
    md: "text-2xl",
    lg: "text-3xl",
    xl: "text-4xl",
};

const colorClasses = {
    positive: "text-emerald-400",
    negative: "text-red-400",
    neutral: "text-white/60",
    default: "text-white/90",
};

export function StatValue({
    value,
    label,
    size = "md",
    variant = "default",
    className,
}: StatValueProps) {
    return (
        <div className={cn("flex flex-col", className)}>
            <span className={cn(
                "font-mono font-bold",
                sizeClasses[size],
                colorClasses[variant]
            )}>
                {value}
            </span>
            {label && (
                <span className="text-[10px] uppercase tracking-wider text-white/40 mt-1">
                    {label}
                </span>
            )}
        </div>
    );
}

// Legacy export for backwards compatibility
export const NeonValue = StatValue;

/* ============================================
   STAT BOX (Minimal)
   ============================================ */

interface GlassStatBoxProps {
    label: string;
    value: string | number;
    subValue?: string;
    trend?: "up" | "down" | "neutral";
    className?: string;
}

export function GlassStatBox({
    label,
    value,
    subValue,
    trend,
    className,
}: GlassStatBoxProps) {
    return (
        <div className={cn(
            "p-3 text-center bg-white/[0.02] rounded-lg border border-white/5",
            className
        )}>
            <div className="text-[9px] text-white/40 uppercase tracking-wide mb-1">
                {label}
            </div>
            <div className={cn(
                "text-lg font-bold font-mono",
                trend === "up" && "text-emerald-400",
                trend === "down" && "text-red-400",
                !trend && "text-white/90"
            )}>
                {value}
            </div>
            {subValue && (
                <div className="text-[10px] text-white/40 mt-0.5">
                    {subValue}
                </div>
            )}
        </div>
    );
}

/* ============================================
   AI INSIGHT (Simple list item)
   ============================================ */

interface AIInsightProps {
    message: string;
    timestamp?: string;
    type?: "insight" | "alert" | "momentum";
    className?: string;
}

const insightBorderColors = {
    insight: "border-l-blue-400",
    alert: "border-l-amber-400",
    momentum: "border-l-purple-400",
};

export function AIInsight({
    message,
    timestamp,
    type = "insight",
    className,
}: AIInsightProps) {
    return (
        <div className={cn(
            "pl-3 border-l-2",
            insightBorderColors[type],
            className
        )}>
            <p className="text-sm text-white/70 leading-relaxed">
                {message}
            </p>
            {timestamp && (
                <span className="text-[10px] text-white/30 mt-1 block">
                    {timestamp}
                </span>
            )}
        </div>
    );
}

/* ============================================
   GLASS DIVIDER (Subtle)
   ============================================ */

interface GlassDividerProps {
    className?: string;
    glow?: boolean;
}

export function GlassDivider({ className, glow = false }: GlassDividerProps) {
    return (
        <div className={cn(
            glow
                ? "h-px bg-gradient-to-r from-transparent via-white/10 to-transparent"
                : "h-px bg-white/5",
            className
        )} />
    );
}

/* ============================================
   CONFIDENCE METER (Clean)
   ============================================ */

interface ConfidenceMeterProps {
    value: number; // 0-100
    label?: string;
    showValue?: boolean;
    className?: string;
}

export function ConfidenceMeter({
    value,
    label,
    showValue = true,
    className,
}: ConfidenceMeterProps) {
    const clampedValue = Math.max(0, Math.min(100, value));

    return (
        <div className={cn("space-y-1.5", className)}>
            {(label || showValue) && (
                <div className="flex items-center justify-between">
                    {label && (
                        <span className="text-[10px] text-white/40 uppercase tracking-wide">
                            {label}
                        </span>
                    )}
                    {showValue && (
                        <span className="text-xs font-mono font-semibold text-white/70">
                            {clampedValue}%
                        </span>
                    )}
                </div>
            )}
            <div className="h-1 bg-white/5 rounded-full overflow-hidden">
                <div
                    className="h-full bg-gradient-to-r from-blue-500 to-emerald-400 rounded-full transition-all duration-500"
                    style={{ width: `${clampedValue}%` }}
                />
            </div>
        </div>
    );
}

/* ============================================
   EDGE INDICATOR (Subtle)
   ============================================ */

interface EdgeIndicatorProps {
    value: number;
    label?: string;
    className?: string;
}

export function EdgeIndicator({
    value,
    label,
    className,
}: EdgeIndicatorProps) {
    const isPositive = value > 0;
    const absValue = Math.abs(value);
    const maxEdge = 10;
    const percentage = Math.min((absValue / maxEdge) * 50, 50);

    return (
        <div className={cn("space-y-2", className)}>
            {label && (
                <div className="flex items-center justify-between">
                    <span className="text-[10px] text-white/40 uppercase tracking-wide">
                        {label}
                    </span>
                    <span className={cn(
                        "text-xs font-mono font-bold",
                        isPositive ? "text-emerald-400" : "text-red-400"
                    )}>
                        {isPositive ? "+" : ""}{value.toFixed(1)}
                    </span>
                </div>
            )}
            <div className="h-1 bg-white/5 rounded-full relative">
                <div className="absolute top-1/2 left-1/2 -translate-y-1/2 w-0.5 h-2 bg-white/20 rounded" />
                {isPositive ? (
                    <div
                        className="absolute top-0 left-1/2 h-full bg-emerald-400 rounded-full"
                        style={{ width: `${percentage}%` }}
                    />
                ) : (
                    <div
                        className="absolute top-0 right-1/2 h-full bg-red-400 rounded-full"
                        style={{ width: `${percentage}%` }}
                    />
                )}
            </div>
        </div>
    );
}

export default GlassCard;
