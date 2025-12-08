'use client';

import { GameDetail } from '@/lib/mock_data';
import { cn } from '@/lib/utils';

interface KeyStatsStripProps {
  game: GameDetail;
}

export function KeyStatsStrip({ game }: KeyStatsStripProps) {
  const { home_stats, away_stats, away_team, home_team } = game;

  if (!home_stats || !away_stats) {
    return null;
  }

  // Parse third down conversion rates
  const parseConversion = (conv: string) => {
    const [made, att] = conv.split('/').map(Number);
    return att > 0 ? Math.round((made / att) * 100) : 0;
  };

  const awayThirdDownPct = parseConversion(away_stats.third_down_conv);
  const homeThirdDownPct = parseConversion(home_stats.third_down_conv);
  const awayRedZonePct = away_stats.red_zone_conv ? parseConversion(away_stats.red_zone_conv) : null;
  const homeRedZonePct = home_stats.red_zone_conv ? parseConversion(home_stats.red_zone_conv) : null;

  const statItems = [
    {
      label: 'Total Yards',
      away: away_stats.total_yards,
      home: home_stats.total_yards,
    },
    {
      label: 'EPA/Play',
      away: away_stats.epa_per_play?.toFixed(2) ?? '\u2014',
      home: home_stats.epa_per_play?.toFixed(2) ?? '\u2014',
      colorize: true,
    },
    {
      label: '3rd Down',
      away: `${awayThirdDownPct}%`,
      home: `${homeThirdDownPct}%`,
    },
    {
      label: 'Red Zone',
      away: awayRedZonePct !== null ? `${awayRedZonePct}%` : '\u2014',
      home: homeRedZonePct !== null ? `${homeRedZonePct}%` : '\u2014',
    },
    {
      label: 'Turnovers',
      away: away_stats.turnovers,
      home: home_stats.turnovers,
      inverse: true,
    },
  ];

  return (
    <div className="bg-white/[0.02] border border-white/[0.06] rounded-xl px-6 py-4">
      <div className="flex items-center justify-between gap-4 overflow-x-auto scrollbar-hide">
        {statItems.map((stat, index) => (
          <div key={stat.label} className="flex items-center gap-4">
            <div className="text-center min-w-[100px]">
              <div className="text-[10px] uppercase tracking-wider text-white/30 mb-1.5">
                {stat.label}
              </div>
              <div className="flex items-center justify-center gap-2 text-sm">
                <span
                  className={cn(
                    'tabular-nums font-medium',
                    stat.colorize && getComparisonColor(stat.away, stat.home, stat.inverse),
                    !stat.colorize && 'text-white/70'
                  )}
                >
                  {stat.away}
                </span>
                <span className="text-white/20">&mdash;</span>
                <span
                  className={cn(
                    'tabular-nums font-medium',
                    stat.colorize && getComparisonColor(stat.home, stat.away, stat.inverse),
                    !stat.colorize && 'text-white/70'
                  )}
                >
                  {stat.home}
                </span>
              </div>
              <div className="text-[10px] text-white/20 mt-1">
                {away_team} / {home_team}
              </div>
            </div>

            {index < statItems.length - 1 && (
              <div className="w-px h-10 bg-white/[0.06] hidden sm:block" />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function getComparisonColor(value: string | number, compareValue: string | number, inverse?: boolean): string {
  const numValue = parseFloat(String(value));
  const numCompare = parseFloat(String(compareValue));

  if (isNaN(numValue) || isNaN(numCompare)) return 'text-white/70';

  const isBetter = inverse ? numValue < numCompare : numValue > numCompare;
  const isWorse = inverse ? numValue > numCompare : numValue < numCompare;

  if (isBetter) return 'text-emerald-400';
  if (isWorse) return 'text-red-400';
  return 'text-white/70';
}

export default KeyStatsStrip;
