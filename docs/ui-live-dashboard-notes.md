# Live Dashboard UI Architecture

## Overview

The Live Dashboard has been refactored into an enterprise-grade glassmorphism design system that emphasizes the AI/model intelligence behind Predictr. The UI is organized into distinct zones with clear visual hierarchy.

---

## Layout Zones

### 1. Top Hero Zone
**Location:** Full-width top section  
**Purpose:** Game state at a glance + AI intelligence summary

Components:
- `HeroGameCard` - Current game score, teams, status, quarter/time
- `AIInsightsSummary` - Model edge, win probability, and primary edge driver
- Live status indicator with animated pulse

### 2. Central Analytics Zone  
**Location:** Main content area (left + center columns)

Components:
- `WinProbabilityCard` - Premium area chart with confidence bands, gradient fills, and glow effects
- `QuarterbackPerformanceCard` - Side-by-side QB capsules with EPA meters and team color accents

### 3. AI Intelligence Rail
**Location:** Right column (sticky/scrolling)

Components:
- `AIIntelligenceRail` - Vertical panel containing:
  - Model confidence band visualization
  - Key insights with timestamps
  - Market inefficiency alerts
  - Momentum shift notifications

### 4. Deep Stats Zone
**Location:** Lower grid section

Components:
- `TeamStatsGrid` - General stats, Downs & Conversions, Negative Plays
- Consistent glass card styling with compact typography

---

## File Entry Points

```
src/app/games/page.tsx          → Main Live dashboard route
src/components/live/
├── LiveDashboard.tsx           → Main dashboard orchestrator
├── LiveLayoutShell.tsx         → Page shell with header
├── HeroGameCard.tsx            → Hero zone game display
├── AIIntelligenceRail.tsx      → Right column AI insights
├── WinProbabilityCard.tsx      → Win probability chart
├── QuarterbackPerformanceCard.tsx → QB performance capsules
├── PreGameLineCard.tsx         → Market vs Model panel
├── ScoringSummaryCard.tsx      → Quarter-by-quarter scoring
├── TeamStatsGrid.tsx           → Stats card grid
├── GameStrip.tsx               → Horizontal game selector
├── DashboardCard.tsx           → Base card primitives
└── index.ts                    → Barrel exports
```

---

## Glassmorphism Primitives

### GlassCard
Base glass surface for all cards.

```tsx
<GlassCard variant="default">...</GlassCard>
<GlassCard variant="elevated">...</GlassCard>
<GlassCard variant="hero">...</GlassCard>
```

**CSS Classes:**
- `.glass-surface` - Base transparent background with blur
- `.glass-surface-elevated` - Stronger elevation and brightness
- `.glass-surface-hero` - Premium variant with subtle gradient border
- `.glass-border` - Subtle light border
- `.glass-glow` - Neon accent glow effect

### Design Tokens (CSS Variables)

```css
/* Glass backgrounds */
--glass-bg: rgba(255, 255, 255, 0.03);
--glass-bg-elevated: rgba(255, 255, 255, 0.05);
--glass-bg-hero: rgba(255, 255, 255, 0.06);

/* Glass borders */
--glass-border: rgba(255, 255, 255, 0.08);
--glass-border-strong: rgba(255, 255, 255, 0.12);

/* Accent colors */
--neon-blue: #00d4ff;
--neon-green: #00ff88;
--neon-purple: #a855f7;

/* Glow effects */
--glow-blue: 0 0 20px rgba(0, 212, 255, 0.3);
--glow-success: 0 0 20px rgba(0, 255, 136, 0.3);
```

---

## Interactive Behaviors

### Hover Effects
- Cards lift slightly (`translateY(-2px)`)
- Border brightens to `--glass-border-strong`
- Subtle glow appears on edges

### State Linking
- Hovering a team highlights related data across cards
- Active game in GameStrip shows selection indicator
- Win probability chart shows tooltip on hover

### Transitions
- All state changes use 200ms ease-out transitions
- Live data updates animate smoothly
- Skeleton loaders have shimmer animation

---

## Component Reuse Guide

### Creating a new glass card:
```tsx
import { GlassCard, GlassCardHeader } from '@/components/ui/glass';

<GlassCard variant="default" className="p-4">
  <GlassCardHeader 
    title="My Section"
    icon={<IconComponent />}
    action={<Button>Action</Button>}
  />
  {/* Content */}
</GlassCard>
```

### Using neon accents:
```tsx
// For positive values (success)
<span className="text-neon-green">+5.2</span>

// For AI/model emphasis
<span className="text-neon-blue">85% Win Prob</span>

// For badges/pills
<span className="bg-neon-blue/20 text-neon-blue px-2 py-1 rounded-full">
  Strong Edge
</span>
```

---

## TODOs / Future Enhancements

- [ ] Command palette (⌘K) for quick navigation
- [ ] Expanded AI insights modal with historical model performance
- [ ] Real-time WebSocket integration for live updates
- [ ] Player hover cards with detailed stats
- [ ] Customizable dashboard layout (drag/drop zones)
- [ ] Light mode glassmorphism variant
- [ ] Mobile-optimized AI rail (collapsible drawer)

---

## Responsive Breakpoints

| Breakpoint | Layout |
|------------|--------|
| < 768px    | Single column, stacked zones, AI rail as drawer |
| 768-1024px | 2 column, AI rail hidden (accessible via toggle) |
| 1024-1280px | 3 column layout, AI rail visible |
| 1280px+    | Full layout with expanded hero zone |

---

## Performance Considerations

1. **Blur effects**: Use `backdrop-blur-sm` (4px) instead of heavy blur for performance
2. **Animations**: All animations use `transform` and `opacity` for GPU acceleration
3. **Charts**: Recharts components wrapped in `ResponsiveContainer` with debounced resize
4. **Skeleton loading**: Shown during initial load and game switches

---

*Last updated: December 2025*

