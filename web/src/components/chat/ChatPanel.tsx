'use client';

import { useRef, useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Send, RotateCcw } from 'lucide-react';
import { useLayout } from '@/contexts/LayoutContext';
import { useAIChat, ChatMessage } from '@/hooks/useAIChat';
import { ChatContext } from '@/lib/ai/context-builder';
import { ChatMessage as ChatMessageComponent } from './ChatMessage';
import { ProcessingIndicator } from './ProcessingIndicator';
import { cn } from '@/lib/utils';
import { useMediaQuery } from '@/hooks/useMediaQuery';

interface ChatPanelProps {
  gameContext?: ChatContext;
}

const QUICK_PROMPTS = [
  { label: 'Why this prediction?', prompt: 'Why does the model favor {favoredTeam}?' },
  { label: "How does it work?", prompt: "How does the prediction model work?" },
  { label: 'Key factors', prompt: 'What are the key factors in this prediction?' },
  { label: 'What to watch', prompt: 'What should I watch for the rest of this game?' },
  { label: 'QB comparison', prompt: 'How do the quarterbacks compare?' },
];

export function ChatPanel({ gameContext }: ChatPanelProps) {
  const { setChatOpen } = useLayout();
  const isMobile = useMediaQuery('(max-width: 768px)');
  const { messages, isLoading, error, sendMessage, clearChat, retryLast } = useAIChat(gameContext);

  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Show processing indicator while waiting for response (hide as soon as streaming starts)
  const lastMessage = messages.length > 0 ? messages[messages.length - 1] : null;
  const shouldShowProcessing = isLoading && lastMessage?.role === 'user';

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  // Focus input on mount
  useEffect(() => {
    setTimeout(() => inputRef.current?.focus(), 100);
  }, []);

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || isLoading) return;

    const message = input;
    setInput('');
    await sendMessage(message);
  };

  const handleQuickPrompt = (prompt: string) => {
    if (!gameContext?.prediction || !gameContext?.game) {
      sendMessage(prompt);
      return;
    }

    const favoredTeam =
      gameContext.prediction.win_prob_home > 0.5
        ? gameContext.game.home_team
        : gameContext.game.away_team;

    const filledPrompt = prompt
      .replace('{favoredTeam}', favoredTeam)
      .replace('{homeTeam}', gameContext.game.home_team)
      .replace('{awayTeam}', gameContext.game.away_team);

    sendMessage(filledPrompt);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  // Mobile renders as bottom sheet overlay
  if (isMobile) {
    return (
      <motion.div
        initial={{ y: '100%' }}
        animate={{ y: 0 }}
        exit={{ y: '100%' }}
        transition={{ type: 'spring', damping: 30, stiffness: 300 }}
        className="fixed inset-0 z-50 bg-[#0a0a0a] flex flex-col"
      >
        {/* Header */}
        <div className="flex-shrink-0 flex items-center justify-between px-5 py-4 border-b border-white/[0.06]">
          <div className="flex items-center gap-3">
            <div>
              <p className="text-[11px] text-white/40">Ask anything about this game</p>
            </div>
          </div>

          <div className="flex items-center gap-1">
            {messages.length > 0 && (
              <button
                onClick={clearChat}
                className="p-2 rounded-lg hover:bg-white/[0.06] text-white/30 hover:text-white/50 transition-colors"
                title="Clear conversation"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
            )}
            <button
              onClick={() => setChatOpen(false)}
              className="p-2 rounded-lg hover:bg-white/[0.06] text-white/30 hover:text-white/50 transition-colors"
              title="Close chat"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto scrollbar-chat">
          {messages.length === 0 ? (
            <EmptyState onQuickPrompt={handleQuickPrompt} />
          ) : (
            <div className="p-4 space-y-4">
              <AnimatePresence initial={false}>
                {messages.map((message, index) => (
                  <ChatMessageComponent
                    key={message.id}
                    message={message}
                    isStreaming={isLoading && index === messages.length - 1 && message.role === 'assistant'}
                  />
                ))}
              </AnimatePresence>

              {/* Processing indicator */}
              <AnimatePresence>
                {shouldShowProcessing && <ProcessingIndicator key="processing" />}
              </AnimatePresence>

              {error && (
                <div className="flex items-center gap-2 text-sm text-red-400/70 px-2">
                  <span>Something went wrong.</span>
                  <button
                    onClick={retryLast}
                    className="text-white/50 hover:text-white/70 underline"
                  >
                    Retry
                  </button>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className="flex-shrink-0 p-4 pt-2 border-t border-white/[0.06]">
          <form onSubmit={handleSubmit} className="flex items-end gap-3">
            <div className="flex-1">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask about the prediction..."
                disabled={isLoading}
                rows={1}
                className="
                  w-full px-4 py-3
                  bg-white/[0.03] hover:bg-white/[0.04]
                  border border-white/[0.08] focus:border-white/[0.15]
                  rounded-xl text-sm text-white/90
                  placeholder:text-white/25
                  focus:outline-none focus:ring-1 focus:ring-white/[0.1]
                  resize-none transition-colors
                  disabled:opacity-50
                "
                style={{ minHeight: '48px', maxHeight: '120px' }}
              />
            </div>
            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className="
                p-3 rounded-xl
                bg-white/[0.06] hover:bg-white/[0.1]
                text-white/40 hover:text-white/60
                disabled:opacity-30 disabled:cursor-not-allowed
                transition-colors
              "
            >
              <Send className="w-4 h-4" />
            </button>
          </form>
        </div>
      </motion.div>
    );
  }

  // Desktop: fixed right panel
  return (
    <div className="flex flex-col h-full bg-[#0a0a0a] border-l border-white/[0.06]">
      {/* Header */}
      <div className="flex-shrink-0 flex items-center justify-between px-5 py-4 border-b border-white/[0.06]">
        <div className="flex items-center gap-3">
          <div>
            <p className="text-[11px] text-white/40">Ask anything about this game</p>
          </div>
        </div>

        <div className="flex items-center gap-1">
          {messages.length > 0 && (
            <button
              onClick={clearChat}
              className="p-2 rounded-lg hover:bg-white/[0.06] text-white/30 hover:text-white/50 transition-colors"
              title="Clear conversation"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          )}
          <button
            onClick={() => setChatOpen(false)}
            className="p-2 rounded-lg hover:bg-white/[0.06] text-white/30 hover:text-white/50 transition-colors"
            title="Close chat"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto scrollbar-chat">
        {messages.length === 0 ? (
          <EmptyState onQuickPrompt={handleQuickPrompt} />
        ) : (
          <div className="p-4 space-y-4">
            <AnimatePresence initial={false}>
              {messages.map((message, index) => (
                <ChatMessageComponent
                  key={message.id}
                  message={message}
                  isStreaming={isLoading && index === messages.length - 1 && message.role === 'assistant'}
                />
              ))}
            </AnimatePresence>

            {/* Processing indicator */}
            <AnimatePresence>
              {shouldShowProcessing && <ProcessingIndicator key="processing" />}
            </AnimatePresence>

            {error && (
              <div className="flex items-center gap-2 text-sm text-red-400/70 px-2">
                <span>Something went wrong.</span>
                <button
                  onClick={retryLast}
                  className="text-white/50 hover:text-white/70 underline"
                >
                  Retry
                </button>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="flex-shrink-0 p-4 pt-2 border-t border-white/[0.06]">
        <form onSubmit={handleSubmit} className="flex items-end gap-3">
          <div className="flex-1">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about the prediction..."
              disabled={isLoading}
              rows={1}
              className="w-full px-4 py-3 bg-white/[0.03] hover:bg-white/[0.04] border border-white/[0.08] focus:border-white/[0.15] rounded-xl text-sm text-white/90 placeholder:text-white/25 focus:outline-none focus:ring-1 focus:ring-white/[0.1] resize-none transition-colors disabled:opacity-50"
              style={{ minHeight: '48px', maxHeight: '120px' }}
            />
          </div>
          <button
            type="submit"
            disabled={!input.trim() || isLoading}
            className="p-3 rounded-xl bg-white/[0.06] hover:bg-white/[0.1] text-white/40 hover:text-white/60 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            <Send className="w-4 h-4" />
          </button>
        </form>
      </div>
    </div>
  );
}

// Empty state with quick prompts
function EmptyState({ onQuickPrompt }: { onQuickPrompt: (prompt: string) => void }) {
  return (
    <div className="h-full flex flex-col items-center justify-center px-6 text-center">
      <h3 className="text-base font-medium text-white/70 mb-2">Ask me anything</h3>
      <p className="text-sm text-white/30 mb-6 max-w-[280px]">
        Model predictions, player analysis, betting insights — I'm here to help.
      </p>

      <div className="flex gap-2 justify-center">
        {QUICK_PROMPTS.slice(0, 3).map((qp) => (
          <button
            key={qp.label}
            onClick={() => onQuickPrompt(qp.prompt)}
            className="px-3.5 py-2 text-xs bg-white/[0.03] hover:bg-white/[0.06] border border-white/[0.06] hover:border-white/[0.1] rounded-lg text-white/50 hover:text-white/70 transition-all duration-200 whitespace-nowrap"
          >
            {qp.label}
          </button>
        ))}
      </div>
    </div>
  );
}

