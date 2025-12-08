'use client';

import { ReactNode } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useLayout } from '@/contexts/LayoutContext';
import { ChatPanel } from '@/components/chat/ChatPanel';
import { ChatToggleButton } from '@/components/chat/ChatToggleButton';
import { ChatContext } from '@/lib/ai/context-builder';

interface DashboardLayoutProps {
  children: ReactNode;
  gameContext?: ChatContext;
}

export function DashboardLayout({ children, gameContext }: DashboardLayoutProps) {
  const { chatOpen } = useLayout();

  return (
    <div className="relative flex h-full w-full flex-1 min-h-0">
      {/* Main Content - Flexible, adapts to chat state, no scrolling */}
      <motion.main
        className="flex-1 overflow-hidden min-h-0 flex"
        animate={{
          marginRight: chatOpen ? 420 : 0,
        }}
        transition={{
          type: 'spring',
          stiffness: 400,
          damping: 40,
        }}
      >
        {children}
      </motion.main>

      {/* Chat Panel - Fixed width, slides in/out */}
      <AnimatePresence mode="wait">
        {chatOpen && (
          <motion.aside
            initial={{ x: 420, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 420, opacity: 0 }}
            transition={{
              type: 'spring',
              stiffness: 400,
              damping: 40,
            }}
            className="fixed right-0 top-12 bottom-0 w-[420px] z-40"
          >
            <ChatPanel gameContext={gameContext} />
          </motion.aside>
        )}
      </AnimatePresence>

      {/* Chat Toggle Button - Visible when chat is closed */}
      <AnimatePresence>
        {!chatOpen && <ChatToggleButton />}
      </AnimatePresence>
    </div>
  );
}

