"use client";

import { forwardRef, ReactNode } from "react";
import { motion, HTMLMotionProps } from "framer-motion";
import { cn } from "@/lib/utils";
import { LucideIcon } from "lucide-react";

/* ============================================
   GLASS CARD VARIANTS
   ============================================ */

type GlassVariant = "default" | "elevated" | "hero" | "rail" | "neon";

interface GlassCardProps extends Omit<HTMLMotionProps<"div">, "children"> {
    children: ReactNode;
    variant?: GlassVariant;
    className?: string;
    noPadding?: boolean;
    interactive?: boolean;
    glowOnHover?: boolean;
}

const variantClasses: Record<GlassVariant, string> = {
    default: "glass-surface",
    elevated: "glass-surface-elevated",
    hero: "glass-surface-hero",
    rail: "glass-rail",
    neon: "glass-card-neon",
};

export const GlassCard = forwardRef<HTMLDivElement, GlassCardProps>(
    ({ 
        children, 
        variant = "default", 
        className, 
        noPadding = false, 
        interactive = false,
        glowOnHover = false,
        ...props 
    }, ref) => {
        return (
            <motion.div
                ref={ref}
                className={cn(
                    variantClasses[variant],
                    !noPadding && "p-4",
                    interactive && "card-interactive cursor-pointer",
                    glowOnHover && "hover-glow-blue",
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
   GLASS CARD HEADER
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
                    <div className="p-2 rounded-lg bg-[var(--glass-bg-elevated)] border border-[var(--glass-border)]">
                        <Icon size={16} className="text-[var(--neon-blue)]" />
                    </div>
                )}
                <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                        <h3 className="text-sm font-semibold text-[var(--foreground)] tracking-tight">
                            {title}
                        </h3>
                        {badge && (
                            <span className={cn("neon-badge", badgeVariantClasses[badge.variant])}>
                                {badge.text}
                            </span>
                        )}
                    </div>
                    {subtitle && (
                        <p className="text-[11px] text-[var(--muted-foreground)] mt-0.5">
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
            "bg-[var(--glass-bg)]",
            "backdrop-blur-md",
            "border-b border-[var(--glass-border)]",
            className
        )}>
            {children}
        </div>
    );
}

/* ============================================
   NEON VALUE DISPLAY
   ============================================ */

interface NeonValueProps {
    value: string | number;
    label?: string;
    size?: "sm" | "md" | "lg" | "xl";
    variant?: "blue" | "green" | "purple" | "default";
    showGlow?: boolean;
    className?: string;
}

const sizeClasses = {
    sm: "text-lg",
    md: "text-2xl",
    lg: "text-3xl",
    xl: "text-4xl",
};

const colorClasses = {
    blue: "text-[var(--neon-blue)]",
    green: "text-[var(--neon-green)]",
    purple: "text-[var(--neon-purple)]",
    default: "text-[var(--foreground)]",
};

const glowClasses = {
    blue: "text-neon-blue",
    green: "text-neon-green",
    purple: "text-neon-purple",
    default: "",
};

export function NeonValue({
    value,
    label,
    size = "md",
    variant = "blue",
    showGlow = true,
    className,
}: NeonValueProps) {
    return (
        <div className={cn("flex flex-col", className)}>
            <span className={cn(
                "hero-number",
                sizeClasses[size],
                showGlow ? glowClasses[variant] : colorClasses[variant]
            )}>
                {value}
            </span>
            {label && (
                <span className="text-[10px] uppercase tracking-wider text-[var(--muted-foreground)] mt-1 font-medium">
                    {label}
                </span>
            )}
        </div>
    );
}

/* ============================================
   GLASS STAT BOX
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
            "glass-inner p-3 text-center",
            className
        )}>
            <div className="text-[9px] text-[var(--muted-foreground)] uppercase font-medium tracking-wide mb-1">
                {label}
            </div>
            <div className={cn(
                "text-lg font-bold font-mono",
                trend === "up" && "text-[var(--neon-green)]",
                trend === "down" && "text-[var(--destructive)]",
                trend === "neutral" && "text-[var(--foreground)]",
                !trend && "text-[var(--foreground)]"
            )}>
                {value}
            </div>
            {subValue && (
                <div className="text-[10px] text-[var(--muted-foreground)] mt-0.5">
                    {subValue}
                </div>
            )}
        </div>
    );
}

/* ============================================
   AI INSIGHT CARD
   ============================================ */

interface AIInsightProps {
    message: string;
    timestamp?: string;
    type?: "insight" | "alert" | "momentum";
    className?: string;
}

const insightTypeStyles = {
    insight: {
        icon: "ai-indicator",
        border: "border-l-[var(--neon-blue)]",
        bg: "bg-[var(--neon-blue-muted)]",
    },
    alert: {
        icon: "",
        border: "border-l-[var(--warning)]",
        bg: "bg-[var(--warning-muted)]",
    },
    momentum: {
        icon: "",
        border: "border-l-[var(--neon-purple)]",
        bg: "bg-[var(--neon-purple-muted)]",
    },
};

export function AIInsight({
    message,
    timestamp,
    type = "insight",
    className,
}: AIInsightProps) {
    const styles = insightTypeStyles[type];
    
    return (
        <div className={cn(
            "glass-inner p-3 border-l-2",
            styles.border,
            className
        )}>
            <div className="flex items-start gap-2">
                {type === "insight" && (
                    <span className={styles.icon} />
                )}
                <div className="flex-1 min-w-0">
                    <p className="ai-insight-text">
                        {message}
                    </p>
                    {timestamp && (
                        <span className="text-[10px] text-[var(--muted-foreground)] mt-1 block">
                            {timestamp}
                        </span>
                    )}
                </div>
            </div>
        </div>
    );
}

/* ============================================
   GLASS DIVIDER
   ============================================ */

interface GlassDividerProps {
    className?: string;
    glow?: boolean;
}

export function GlassDivider({ className, glow = false }: GlassDividerProps) {
    return (
        <div className={cn(
            glow ? "section-divider" : "h-px bg-[var(--glass-border)]",
            className
        )} />
    );
}

/* ============================================
   CONFIDENCE METER
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
                        <span className="text-[10px] text-[var(--muted-foreground)] uppercase tracking-wide font-medium">
                            {label}
                        </span>
                    )}
                    {showValue && (
                        <span className="text-xs font-mono font-semibold text-[var(--neon-blue)]">
                            {clampedValue}%
                        </span>
                    )}
                </div>
            )}
            <div className="confidence-meter">
                <div 
                    className="confidence-meter-fill"
                    style={{ width: `${clampedValue}%` }}
                />
            </div>
        </div>
    );
}

/* ============================================
   EDGE INDICATOR
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
    const maxEdge = 10; // Max edge for visualization
    const percentage = Math.min((absValue / maxEdge) * 50, 50);
    
    return (
        <div className={cn("space-y-2", className)}>
            {label && (
                <div className="flex items-center justify-between">
                    <span className="text-[10px] text-[var(--muted-foreground)] uppercase tracking-wide font-medium">
                        {label}
                    </span>
                    <span className={cn(
                        "text-xs font-mono font-bold",
                        isPositive ? "text-[var(--neon-green)]" : "text-[var(--destructive)]"
                    )}>
                        {isPositive ? "+" : ""}{value.toFixed(1)}
                    </span>
                </div>
            )}
            <div className="edge-bar">
                <div 
                    className="absolute top-1/2 left-1/2 -translate-y-1/2 w-0.5 h-3 bg-[var(--glass-border-strong)] rounded"
                />
                {isPositive ? (
                    <div 
                        className="edge-bar-positive"
                        style={{ width: `${percentage}%` }}
                    />
                ) : (
                    <div 
                        className="edge-bar-negative"
                        style={{ width: `${percentage}%` }}
                    />
                )}
            </div>
        </div>
    );
}

export default GlassCard;

