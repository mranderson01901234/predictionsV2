'use client';

import { GameDetail, Quarterback } from '@/lib/mock_data';
import { cn } from '@/lib/utils';
import { getQBImage, getQBInitials } from '@/lib/data/qb-images';

interface QBComparisonProps {
  game: GameDetail;
}

export function QBComparison({ game }: QBComparisonProps) {
  const { home_qb, away_qb, away_team, home_team } = game;

  if (!home_qb || !away_qb) {
    return (
      <div className="bg-white/[0.02] border border-white/[0.06] rounded-2xl p-8 h-full flex items-center justify-center">
        <p className="text-white/30 text-sm">QB stats unavailable</p>
      </div>
    );
  }

  return (
    <div className="bg-white/[0.02] border border-white/[0.06] rounded-2xl overflow-hidden h-full flex flex-col">
      {/* QB Cards Container */}
      <div className="flex-1 p-4 space-y-4 overflow-y-auto scrollbar-thin min-h-0">
        <QBCard qb={away_qb} team={away_team} />
        <QBCard qb={home_qb} team={home_team} />
      </div>
    </div>
  );
}

function QBCard({ qb, team }: { qb: Quarterback; team: string }) {
  // Try to get image from our utility first, fallback to headshot_url from data
  const imageUrl = getQBImage(qb.name) || qb.headshot_url;
  const initials = getQBInitials(qb.name);
  const epa = qb.epa ?? 0;
  const epaPositive = epa > 0;
  const completionPct = qb.attempts > 0
    ? Math.round((qb.completions / qb.attempts) * 100)
    : 0;

  return (
    <div className="relative bg-gradient-to-br from-white/[0.04] to-white/[0.01] rounded-2xl overflow-hidden">
      {/* Ambient glow effect */}
      <div className="absolute -top-20 -right-20 w-40 h-40 bg-white/[0.02] rounded-full blur-3xl pointer-events-none" />

      <div className="relative p-5">
        {/* Main Content Row */}
        <div className="flex items-start gap-4">

          {/* Headshot */}
          <div className="flex-shrink-0 relative">
            {/* Shadow blur underneath */}
            <div
              className="absolute inset-0 rounded-2xl bg-black/30 blur-xl"
              style={{ transform: 'translateY(6px) scale(0.9)' }}
            />

            {imageUrl ? (
              <img
                src={imageUrl}
                alt={qb.name}
                className="relative w-32 h-32 object-cover object-top rounded-2xl"
                onError={(e) => {
                  (e.target as HTMLImageElement).style.display = 'none';
                  const fallback = (e.target as HTMLImageElement).nextElementSibling;
                  if (fallback) (fallback as HTMLElement).style.display = 'flex';
                }}
              />
            ) : null}
            {/* Fallback initials */}
            <div
              className={cn(
                "relative w-32 h-32 rounded-2xl bg-white/[0.06] items-center justify-center",
                imageUrl ? "hidden" : "flex"
              )}
            >
              <span className="text-4xl font-bold text-white/40">
                {initials}
              </span>
            </div>
          </div>

          {/* Info Column */}
          <div className="flex-1 min-w-0 py-0.5">
            {/* Name & Team */}
            <h4 className="text-lg font-bold text-white truncate mb-0.5">
              {qb.name}
            </h4>
            <p className="text-xs text-white/40 mb-2.5">
              {team} &bull; Quarterback
            </p>

            {/* EPA Badge */}
            <div className="inline-flex items-center gap-2 px-2.5 py-1.5 rounded-lg bg-white/[0.04]">
              <span className="text-[10px] text-white/40 uppercase tracking-wide">EPA</span>
              <span
                className={cn(
                  'text-sm font-bold tabular-nums',
                  epaPositive ? 'text-emerald-400' : 'text-red-400'
                )}
              >
                {epaPositive ? '+' : ''}{epa.toFixed(1)}
              </span>
            </div>
          </div>

          {/* QBR - Hero Number */}
          <div className="flex-shrink-0 text-right">
            <div className="text-4xl font-bold tabular-nums text-white tracking-tight">
              {qb.qbr?.toFixed(1) ?? '\u2014'}
            </div>
            <div className="text-[10px] text-white/40 uppercase tracking-widest mt-0.5">
              QBR
            </div>
          </div>
        </div>

        {/* Stats Row */}
        <div className="mt-4 pt-4 border-t border-white/[0.06]">
          <div className="grid grid-cols-5 gap-1.5">
            <StatPill label="C/ATT" value={`${qb.completions}/${qb.attempts}`} />
            <StatPill label="YDS" value={qb.yards} />
            <StatPill
              label="TD"
              value={qb.tds}
              variant={qb.tds > 0 ? 'positive' : 'neutral'}
            />
            <StatPill
              label="INT"
              value={qb.ints}
              variant={qb.ints > 0 ? 'negative' : 'neutral'}
            />
            <StatPill label="CMP%" value={`${completionPct}%`} />
          </div>
        </div>
      </div>
    </div>
  );
}

function StatPill({
  label,
  value,
  variant = 'neutral',
}: {
  label: string;
  value: string | number;
  variant?: 'positive' | 'negative' | 'neutral';
}) {
  return (
    <div className="text-center py-2 px-1.5 rounded-xl bg-white/[0.03] border border-white/[0.04]">
      <div className="text-[9px] text-white/30 uppercase tracking-wider mb-0.5">
        {label}
      </div>
      <div
        className={cn(
          'text-xs font-semibold tabular-nums',
          variant === 'positive' && 'text-emerald-400',
          variant === 'negative' && 'text-red-400',
          variant === 'neutral' && 'text-white/80'
        )}
      >
        {value}
      </div>
    </div>
  );
}

export default QBComparison;
