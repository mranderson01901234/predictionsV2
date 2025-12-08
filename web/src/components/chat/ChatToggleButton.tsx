'use client';

import { motion } from 'framer-motion';
import { useLayout } from '@/contexts/LayoutContext';

export function ChatToggleButton() {
  const { setChatOpen } = useLayout();

  return (
    <motion.button
      initial={{ opacity: 0, scale: 0.9, y: 20 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.9, y: 20 }}
      transition={{
        type: 'spring',
        stiffness: 400,
        damping: 30,
        delay: 0.1,
      }}
      onClick={() => setChatOpen(true)}
      className="fixed bottom-6 right-6 z-50 flex items-center gap-2.5 px-4 py-3 bg-[#0a0a0a]/90 hover:bg-[#111111] border border-white/[0.08] hover:border-white/[0.15] rounded-xl backdrop-blur-md shadow-lg shadow-black/20 transition-all duration-200 group cursor-pointer"
    >
      {/* Label */}
      <span className="text-sm font-medium text-white/40 group-hover:text-white/60 transition-colors">
        Chat
      </span>
    </motion.button>
  );
}

