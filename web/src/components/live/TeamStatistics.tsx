'use client';

import { useState } from 'react';
import { GameDetail, TeamStats } from '@/lib/mock_data';
import { cn } from '@/lib/utils';

interface TeamStatisticsProps {
  game: GameDetail;
}

const STAT_CATEGORIES = [
  { id: 'general', label: 'General' },
  { id: 'passing', label: 'Passing' },
  { id: 'rushing', label: 'Rushing' },
  { id: 'conversions', label: 'Conversions' },
  { id: 'negative', label: 'Negative Plays' },
];

export function TeamStatistics({ game }: TeamStatisticsProps) {
  const [activeCategory, setActiveCategory] = useState('general');
  const { home_stats, away_stats, away_team, home_team } = game;

  if (!home_stats || !away_stats) {
    return (
      <div className="bg-white/[0.02] border border-white/[0.06] rounded-xl p-6 h-full">
        <div className="text-white/30 text-sm text-center">Stats unavailable</div>
      </div>
    );
  }

  return (
    <div className="bg-white/[0.02] border border-white/[0.06] rounded-xl overflow-hidden h-full flex flex-col">
      {/* Category Tabs */}
      <div className="flex items-center gap-1.5 px-4 py-3 border-b border-white/[0.06] overflow-x-auto scrollbar-hide flex-shrink-0">
        {STAT_CATEGORIES.map((cat) => (
          <button
            key={cat.id}
            onClick={() => setActiveCategory(cat.id)}
            className={cn(
              'px-4 py-2.5 text-base font-medium rounded-lg whitespace-nowrap transition-colors',
              activeCategory === cat.id
                ? 'bg-white/[0.08] text-white/90'
                : 'text-white/40 hover:text-white/60 hover:bg-white/[0.04]'
            )}
          >
            {cat.label}
          </button>
        ))}
      </div>

      {/* Stats Table */}
      <div className="flex-1 p-4 overflow-y-auto scrollbar-thin min-h-0">
        <StatsTable
          category={activeCategory}
          awayStats={away_stats}
          homeStats={home_stats}
          awayTeam={away_team}
          homeTeam={home_team}
        />
      </div>
    </div>
  );
}

function StatsTable({
  category,
  awayStats,
  homeStats,
  awayTeam,
  homeTeam,
}: {
  category: string;
  awayStats: TeamStats;
  homeStats: TeamStats;
  awayTeam: string;
  homeTeam: string;
}) {
  const rows = getStatsForCategory(category, awayStats, homeStats);

  return (
    <table className="w-full">
      <thead>
        <tr className="text-sm uppercase tracking-wider text-white/30">
          <th className="text-left pb-4 font-medium">Stat</th>
          <th className="text-right pb-4 font-medium w-32">{awayTeam}</th>
          <th className="text-right pb-4 font-medium w-32">{homeTeam}</th>
        </tr>
      </thead>
      <tbody className="text-lg">
        {rows.map((row, index) => (
          <tr
            key={row.label}
            className={cn(index !== rows.length - 1 && 'border-b border-white/[0.04]')}
          >
            <td className="py-4 text-white/50">{row.label}</td>
            <td
              className={cn(
                'py-4 text-right tabular-nums font-semibold',
                row.colorize ? getValueColor(row.away) : 'text-white/80'
              )}
            >
              {row.away}
            </td>
            <td
              className={cn(
                'py-4 text-right tabular-nums font-semibold',
                row.colorize ? getValueColor(row.home) : 'text-white/80'
              )}
            >
              {row.home}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function getStatsForCategory(
  category: string,
  away: TeamStats,
  home: TeamStats
): Array<{ label: string; away: string | number; home: string | number; colorize?: boolean }> {
  switch (category) {
    case 'general':
      return [
        { label: 'Total Yards', away: away.total_yards ?? '\u2014', home: home.total_yards ?? '\u2014' },
        { label: 'Total Plays', away: away.plays ?? '\u2014', home: home.plays ?? '\u2014' },
        { label: 'Yards / Play', away: away.yards_per_play?.toFixed(1) ?? '\u2014', home: home.yards_per_play?.toFixed(1) ?? '\u2014' },
        { label: 'EPA / Play', away: away.epa_per_play?.toFixed(2) ?? '\u2014', home: home.epa_per_play?.toFixed(2) ?? '\u2014', colorize: true },
        { label: 'Time of Possession', away: away.possession ?? '\u2014', home: home.possession ?? '\u2014' },
      ];

    case 'passing':
      return [
        { label: 'Pass Yards', away: away.passing_yards ?? '\u2014', home: home.passing_yards ?? '\u2014' },
        { label: 'Comp / Att', away: away.comp_att ?? '\u2014', home: home.comp_att ?? '\u2014' },
        { label: 'Yards / Pass', away: away.yards_per_pass?.toFixed(1) ?? '\u2014', home: home.yards_per_pass?.toFixed(1) ?? '\u2014' },
        { label: 'Pass EPA', away: away.epa_per_pass?.toFixed(2) ?? '\u2014', home: home.epa_per_pass?.toFixed(2) ?? '\u2014', colorize: true },
        { label: 'Pass First Downs', away: away.passing_first_downs ?? '\u2014', home: home.passing_first_downs ?? '\u2014' },
      ];

    case 'rushing':
      return [
        { label: 'Rush Yards', away: away.rushing_yards ?? '\u2014', home: home.rushing_yards ?? '\u2014' },
        { label: 'Attempts', away: away.rush_attempts ?? '\u2014', home: home.rush_attempts ?? '\u2014' },
        { label: 'Yards / Rush', away: away.yards_per_rush?.toFixed(1) ?? '\u2014', home: home.yards_per_rush?.toFixed(1) ?? '\u2014' },
        { label: 'Rush EPA', away: away.epa_per_rush?.toFixed(2) ?? '\u2014', home: home.epa_per_rush?.toFixed(2) ?? '\u2014', colorize: true },
        { label: 'Rush First Downs', away: away.rushing_first_downs ?? '\u2014', home: home.rushing_first_downs ?? '\u2014' },
      ];

    case 'conversions':
      return [
        { label: '1st Downs', away: away.first_downs ?? '\u2014', home: home.first_downs ?? '\u2014' },
        { label: 'Passing 1st Downs', away: away.passing_first_downs ?? '\u2014', home: home.passing_first_downs ?? '\u2014' },
        { label: 'Rushing 1st Downs', away: away.rushing_first_downs ?? '\u2014', home: home.rushing_first_downs ?? '\u2014' },
        { label: 'Penalty 1st Downs', away: away.penalty_first_downs ?? '\u2014', home: home.penalty_first_downs ?? '\u2014' },
        { label: '3rd Down', away: away.third_down_conv ?? '\u2014', home: home.third_down_conv ?? '\u2014' },
        { label: 'Red Zone', away: away.red_zone_conv ?? '\u2014', home: home.red_zone_conv ?? '\u2014' },
      ];

    case 'negative':
      return [
        { label: 'Turnovers', away: away.turnovers ?? 0, home: home.turnovers ?? 0 },
        { label: 'Sacks Allowed', away: away.sacks ?? 0, home: home.sacks ?? 0 },
        { label: 'Sack Yards Lost', away: away.sack_yards ?? 0, home: home.sack_yards ?? 0 },
        { label: 'Penalties', away: away.penalties ?? 0, home: home.penalties ?? 0 },
        { label: 'Penalty Yards', away: away.penalty_yards ?? 0, home: home.penalty_yards ?? 0 },
      ];

    default:
      return [];
  }
}

function getValueColor(value: string | number): string {
  const num = parseFloat(String(value));
  if (isNaN(num)) return 'text-white/80';
  if (num > 0) return 'text-emerald-400';
  if (num < 0) return 'text-red-400';
  return 'text-white/80';
}

export default TeamStatistics;
