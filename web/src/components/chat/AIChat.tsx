"use client";

import { useEffect, useRef } from "react";
import { GameDetail } from "@/lib/mock_data";
import { useAIChat, ChatMessage } from "@/hooks/useAIChat";
import { ChatMessage as ChatMessageComponent } from "./ChatMessage";
import { ChatInput } from "./ChatInput";
import { QuickPrompts } from "./QuickPrompts";
import { ProcessingIndicator } from "./ProcessingIndicator";
import { cn } from "@/lib/utils";
import { motion, AnimatePresence } from "framer-motion";
import { X, AlertCircle } from "lucide-react";
import { ChatContext } from "@/lib/ai/context-builder";

interface AIChatProps {
  game?: GameDetail;
  gameContext?: ChatContext;
  onClose?: () => void;
  className?: string;
}

export function AIChat({ game, gameContext, onClose, className }: AIChatProps) {
  // Use gameContext if provided, otherwise fall back to game
  const context = gameContext || game;
  const { messages, isLoading, error, sendMessage, clearChat, retryLast } = useAIChat(context);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const handleQuickPrompt = (prompt: string) => {
    sendMessage(prompt);
  };

  // Show processing indicator while waiting for response (hide as soon as streaming starts)
  const lastMessage = messages.length > 0 ? messages[messages.length - 1] : null;
  const shouldShowProcessing = isLoading && lastMessage?.role === "user";

  return (
    <motion.div
      ref={containerRef}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 20 }}
      className={cn(
        "flex flex-col h-full",
        "bg-white/[0.02] border border-white/[0.06] rounded-xl",
        "backdrop-blur-sm",
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-white/[0.06] flex-shrink-0">
        <div className="flex items-center gap-2">
        </div>
        <div className="flex items-center gap-2">
          {messages.length > 0 && (
            <button
              onClick={clearChat}
              className="text-xs text-white/50 hover:text-white/70 transition-colors"
            >
              Clear
            </button>
          )}
          {onClose && (
            <button
              onClick={onClose}
              className="p-1 rounded-lg hover:bg-white/[0.05] transition-colors"
            >
              <X size={14} className="text-white/50" />
            </button>
          )}
        </div>
      </div>

      {/* Error Banner */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="px-4 py-2 bg-red-500/10 border-b border-red-500/20 flex items-center gap-2"
          >
            <AlertCircle size={14} className="text-red-400 flex-shrink-0" />
            <p className="text-xs text-red-400">{error}</p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto scrollbar-chat px-4 py-4 min-h-0">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center px-4">
            <p className="text-sm text-white/50 mb-4">
              Ask me about the model, game analysis, or your bet
            </p>
            <QuickPrompts game={game || gameContext?.game} onSelect={handleQuickPrompt} />
          </div>
        ) : (
          <>
            {messages.map((message) => (
              <ChatMessageComponent
                key={message.id}
                message={message}
                isStreaming={false}
              />
            ))}
            {/* Show processing indicator when loading */}
            <AnimatePresence>
              {shouldShowProcessing && (
                <ProcessingIndicator key="processing-indicator" />
              )}
            </AnimatePresence>
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* Quick Prompts (when messages exist) */}
      {messages.length > 0 && (
        <div className="px-4 pb-2 flex-shrink-0">
          <QuickPrompts game={game || gameContext?.game} onSelect={handleQuickPrompt} />
        </div>
      )}

      {/* Input */}
      <div className="px-4 pb-4 pt-2 border-t border-white/[0.06] flex-shrink-0">
        <ChatInput onSend={sendMessage} isLoading={isLoading} disabled={!game && !gameContext} />
      </div>
    </motion.div>
  );
}

