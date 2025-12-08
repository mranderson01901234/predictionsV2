# Live Dashboard UI Architecture

> **Last Updated**: December 2024  
> **Version**: 2.0 - Enterprise Glassmorphism Redesign

## Overview

The Live Dashboard is the primary real-time analytics view for the Predictr platform. It displays NFL game data with AI-powered predictions, win probability charts, QB performance metrics, and detailed team statistics.

---

## 📁 File Structure

```
src/components/live/
├── index.ts                    # Barrel exports
├── LiveDashboard.tsx           # Main orchestrator component
├── LiveLayoutShell.tsx         # Page shell with header
├── GameStrip.tsx               # Horizontal game selector
├── HeroGameCard.tsx            # Hero zone: game info + AI summary
├── WinProbabilityCard.tsx      # Win probability chart
├── QuarterbackPerformanceCard.tsx  # QB capsules
├── PreGameLineCard.tsx         # Market vs Model lines
├── ScoringSummaryCard.tsx      # Score by quarter
├── TeamStatsGrid.tsx           # Stats tables (General/Downs/Negative)
├── AIIntelligenceRail.tsx      # AI Co-Pilot sidebar rail
└── DashboardCard.tsx           # Base card + skeletons
```

---

## 🎨 Layout Zones

The dashboard is organized into distinct visual zones with clear hierarchy:

### Zone 1: Hero (Top)
**Component**: `HeroGameCard`  
**Purpose**: Immediate game context + AI prediction summary at a glance

- Current game score, teams, status
- Model win probability, edge, confidence
- AI summary sentence with edge classification badge

### Zone 2: Analytics (Middle)
**Components**: `WinProbabilityCard` + `QuarterbackPerformanceCard`  
**Purpose**: Primary analytics data visualizations

- Win probability time-series chart with glow effects
- QB performance capsules with EPA bars and stats

### Zone 3: AI Intelligence Rail (Right)
**Component**: `AIIntelligenceRail`  
**Purpose**: AI Co-Pilot insights and alerts (sticky sidebar)

- Model confidence meter
- Edge analysis (spread + total)
- Real-time alerts
- Key insights with timestamps
- Live indicator when game in progress

### Zone 4: Deep Stats (Bottom)
**Components**: `TeamStatsGrid` (General, Downs, Negative plays)  
**Purpose**: Detailed statistical breakdown

- Three-card grid on desktop, tabbed on mobile
- Comparison rows with EPA coloring

---

## 🔮 Glassmorphism Design System

### CSS Variables (from globals.css)

```css
/* Glass backgrounds */
--glass-bg: rgba(255, 255, 255, 0.02);
--glass-bg-subtle: rgba(255, 255, 255, 0.03);
--glass-bg-medium: rgba(255, 255, 255, 0.05);
--glass-bg-elevated: rgba(255, 255, 255, 0.07);
--glass-bg-hero: rgba(255, 255, 255, 0.08);

/* Glass borders */
--glass-border: rgba(255, 255, 255, 0.06);
--glass-border-medium: rgba(255, 255, 255, 0.10);
--glass-border-strong: rgba(255, 255, 255, 0.15);
--glass-border-neon: rgba(0, 212, 255, 0.3);

/* Neon accents */
--neon-blue: #00d4ff;
--neon-green: #00ff88;
--neon-purple: #a855f7;

/* Glow effects */
--glow-blue: 0 0 20px var(--neon-blue-glow);
--glow-green: 0 0 20px var(--neon-green-glow);
```

### CSS Classes

| Class | Usage |
|-------|-------|
| `.glass-surface` | Standard card with subtle glass |
| `.glass-surface-elevated` | Higher elevation glass |
| `.glass-surface-hero` | Hero-level premium glass with gradient border |
| `.glass-rail` | Sidebar rail glass |
| `.glass-card-neon` | Neon-bordered glass card |
| `.glass-inner` | Nested glass elements |
| `.dashboard-card` | Standard dashboard card with hover |

### React Components

```tsx
// src/components/ui/glass.tsx
import { GlassCard } from '@/components/ui/glass';

// Variants: default | elevated | hero | rail | neon
<GlassCard variant="elevated" interactive glowOnHover>
  {children}
</GlassCard>

// Additional exports:
// - GlassCardHeader
// - GlassPanel
// - NeonValue
// - GlassStatBox
// - AIInsight
// - GlassDivider
// - ConfidenceMeter
// - EdgeIndicator
```

---

## 📊 Component Specifications

### WinProbabilityCard
- **Chart Library**: Recharts (AreaChart)
- **Features**:
  - Gradient fill under line
  - 50% reference line (dashed)
  - Custom tooltip with glass styling
  - Animated dot for current state
  - Glow filter on line

### QuarterbackPerformanceCard
- **Layout**: Two columns (away | home)
- **Metrics**: Yards, TD, INT, EPA total, EPA/play, completion %, QBR
- **Visual**: EPA progress bars with positive/negative coloring

### PreGameLineCard (Market vs Model)
- **Layout**: Three-column comparison (Opening | Current | Model)
- **Features**: Edge indicator, favored team badge
- **Styling**: Highlight model column when edge significant

### AIIntelligenceRail
- **Position**: Sticky right column
- **Sections**:
  1. Model Confidence (meter + volatility/certainty)
  2. Edge Analysis (spread + total bars)
  3. Alerts (severity-coded)
  4. Key Insights (typed: insight/alert/momentum)
  5. Live status indicator

---

## 🎬 Animation System

### Framer Motion Patterns

```tsx
// Card entrance
initial={{ opacity: 0, y: 12 }}
animate={{ opacity: 1, y: 0 }}
transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}

// Page transitions
<AnimatePresence mode="wait">
  <motion.div key={selectedGameId} ...>
</AnimatePresence>
```

### CSS Animations

| Class | Effect |
|-------|--------|
| `.animate-fade-in` | Fade + slide up |
| `.animate-zone-reveal` | Scale + blur reveal |
| `.animate-glow-pulse` | Pulsing glow effect |
| `.animate-data-update` | Flash on data change |
| `.stagger-fade-in > *` | Staggered child animations |
| `.live-indicator` | Pulsing live dot |
| `.ai-scan-effect` | Scanning line effect |

---

## 📱 Responsive Breakpoints

| Breakpoint | Layout Changes |
|------------|----------------|
| < 1024px | AI Rail moves below analytics, stats become tabbed |
| < 768px | Single column layout, game strip scrolls |
| 1024px+ | Full 3-zone grid layout |
| 1280px+ | Max content width enforced |

---

## 🔌 Data Flow

```
GamesPage (server component)
└── LiveDashboard (client)
    ├── games: Game[]
    └── initialDetails: Record<string, GameDetail>
        
GameDetail includes:
├── scoring_summary
├── win_probability[]
├── home_stats / away_stats
├── home_qb / away_qb
├── market?: MarketSnapshot
└── prediction?: Prediction
```

---

## 🚀 Future Enhancements (TODOs)

- [ ] Command Palette (⌘K) for quick navigation
- [ ] Cross-component hover linking (highlight team data across cards)
- [ ] Real-time WebSocket data updates
- [ ] Expanded AI insights with momentum shifts
- [ ] Historical win probability comparison overlay
- [ ] Mobile-optimized QB capsule layouts

---

## 📚 Related Files

- `src/app/games/page.tsx` - Route entry point
- `src/lib/mock_data.ts` - Data types and mock data
- `src/app/globals.css` - Design tokens and utilities
- `src/components/ui/glass.tsx` - Glass component library

