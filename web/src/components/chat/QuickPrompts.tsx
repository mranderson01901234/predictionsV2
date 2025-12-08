"use client";

import { GameDetail } from "@/lib/mock_data";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

interface QuickPrompt {
  label: string;
  prompt: string;
}

interface QuickPromptsProps {
  game?: GameDetail;
  onSelect: (prompt: string) => void;
  className?: string;
}

export function QuickPrompts({ game, onSelect, className }: QuickPromptsProps) {
  if (!game) return null;

  const prediction = game.prediction;
  const market = game.market;
  const modelFavorsHome = prediction ? prediction.predicted_spread < 0 : false;
  const favoredTeam = modelFavorsHome ? game.home_team : game.away_team;
  const underdogTeam = modelFavorsHome ? game.away_team : game.home_team;
  const spread = market?.spread_home || 0;

  const prompts: QuickPrompt[] = [
    {
      label: "Why this prediction?",
      prompt: `Why does the model favor ${favoredTeam} in this game?`,
    },
    {
      label: "How does it work?",
      prompt: "How does the prediction model work?",
    },
    {
      label: "Key factors",
      prompt: "What are the key factors driving the model's prediction right now?",
    },
    {
      label: "What to watch",
      prompt: "What should I watch for in the rest of this game?",
    },
  ];

  return (
    <div className={cn("flex gap-2 justify-center", className)}>
      {prompts.slice(0, 3).map((prompt, index) => (
        <motion.button
          key={prompt.label}
          onClick={() => onSelect(prompt.prompt)}
          className={cn(
            "px-3 py-1.5 text-xs font-medium rounded-lg",
            "bg-white/[0.03] hover:bg-white/[0.06]",
            "border border-white/[0.06] hover:border-white/[0.10]",
            "text-white/70 hover:text-white/90",
            "transition-colors duration-200",
            "backdrop-blur-sm"
          )}
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.05 }}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          {prompt.label}
        </motion.button>
      ))}
    </div>
  );
}

