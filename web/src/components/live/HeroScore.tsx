'use client';

import { GameDetail } from '@/lib/mock_data';
import { cn } from '@/lib/utils';
import { motion } from 'framer-motion';

interface HeroScoreProps {
  game: GameDetail;
}

export function HeroScore({ game }: HeroScoreProps) {
  const {
    home_team,
    away_team,
    home_score,
    away_score,
    quarter,
    time_remaining,
    status,
    home_record,
    away_record,
    prediction,
    market,
  } = game;

  const isLive = status === 'Live';
  const isFinal = status === 'Final';

  // Calculate edge: difference between model spread and market spread
  const edge = prediction && market
    ? prediction.predicted_spread - market.spread_home
    : prediction?.edge_spread ?? null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      className="bg-white/[0.02] border border-white/[0.06] rounded-xl p-5"
    >
        {/* Live Badge */}
        {isLive && (
          <div className="flex items-center justify-center gap-2 mb-4">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
            </span>
            <span className="text-xs font-medium text-emerald-400 uppercase tracking-wider">Live</span>
            {quarter && time_remaining && (
              <span className="text-xs text-white/40 ml-2">
                Q{quarter} &bull; {time_remaining}
              </span>
            )}
          </div>
        )}

        {/* Final Badge */}
        {isFinal && (
          <div className="flex items-center justify-center mb-4">
            <span className="text-xs font-medium text-white/40 uppercase tracking-wider">Final</span>
          </div>
        )}

        {/* Scheduled Time */}
        {!isLive && !isFinal && (
          <div className="flex items-center justify-center mb-4">
            <span className="text-xs font-medium text-white/40">
              {new Date(game.date).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}
            </span>
          </div>
        )}

        {/* Score Display */}
        <div className="flex items-center justify-center px-4 md:px-8">
          {/* Scores in Center */}
          <div className="flex items-center gap-6 md:gap-10">
            {/* Away Score */}
            <div className="text-center">
              <div className={cn(
                "text-4xl md:text-6xl font-bold tabular-nums",
                away_score > home_score ? "text-white/90" : "text-white/40"
              )}>
                {away_score}
              </div>
              <div className="text-sm text-white/50 font-medium mt-1">{away_team}</div>
              {away_record && (
                <div className="text-xs text-white/30">{away_record}</div>
              )}
            </div>

            {/* Separator */}
            <span className="text-3xl text-white/20 font-light">&mdash;</span>

            {/* Home Score */}
            <div className="text-center">
              <div className={cn(
                "text-4xl md:text-6xl font-bold tabular-nums",
                home_score > away_score ? "text-white/90" : "text-white/40"
              )}>
                {home_score}
              </div>
              <div className="text-sm text-white/50 font-medium mt-1">{home_team}</div>
              {home_record && (
                <div className="text-xs text-white/30">{home_record}</div>
              )}
            </div>
          </div>
        </div>

        {/* Model Prediction Bar */}
        {prediction && (
          <div className="flex flex-wrap items-center justify-center gap-3 md:gap-6 mt-5 pt-4 border-t border-white/[0.06]">
            <PredictionStat
              label="Model"
              value={`${prediction.predicted_spread > 0 ? '+' : ''}${prediction.predicted_spread.toFixed(1)}`}
            />

            <div className="hidden md:block w-px h-8 bg-white/[0.08]" />

            <PredictionStat
              label="Market"
              value={
                market?.spread_home !== null && market?.spread_home !== undefined
                  ? `${market.spread_home > 0 ? '+' : ''}${market.spread_home}`
                  : '\u2014'
              }
            />

            <div className="hidden md:block w-px h-8 bg-white/[0.08]" />

            <PredictionStat
              label="Edge"
              value={edge !== null ? `${edge > 0 ? '+' : ''}${edge.toFixed(1)}` : '\u2014'}
              highlight={edge !== null ? (edge > 0 ? 'positive' : edge < 0 ? 'negative' : undefined) : undefined}
            />

            <div className="hidden md:block w-px h-8 bg-white/[0.08]" />

            <PredictionStat
              label="Win Prob"
              value={`${(prediction.win_prob_home * 100).toFixed(0)}%`}
              subtext={home_team}
            />

            {prediction.confidence_score && (
              <>
                <div className="hidden md:block w-px h-8 bg-white/[0.08]" />
                <PredictionStat
                  label="Confidence"
                  value={`${prediction.confidence_score.toFixed(0)}%`}
                />
              </>
            )}
          </div>
        )}
    </motion.div>
  );
}

function PredictionStat({
  label,
  value,
  subtext,
  highlight,
}: {
  label: string;
  value: string;
  subtext?: string;
  highlight?: 'positive' | 'negative';
}) {
  return (
    <div className="text-center min-w-[80px]">
      <div className="text-[10px] uppercase tracking-wider text-white/30 mb-1">
        {label}
      </div>
      <div
        className={cn(
          'text-lg font-semibold tabular-nums',
          highlight === 'positive' && 'text-emerald-400',
          highlight === 'negative' && 'text-red-400',
          !highlight && 'text-white/80'
        )}
      >
        {value}
      </div>
      {subtext && (
        <div className="text-[10px] text-white/30 mt-0.5">{subtext}</div>
      )}
    </div>
  );
}

export default HeroScore;
