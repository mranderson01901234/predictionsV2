"use client";

import { useState, useRef, useEffect } from "react";
import { Send, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";

interface ChatInputProps {
  onSend: (message: string) => void;
  isLoading: boolean;
  disabled?: boolean;
}

export function ChatInput({ onSend, isLoading, disabled }: ChatInputProps) {
  const [message, setMessage] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 120)}px`;
    }
  }, [message]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() && !isLoading && !disabled) {
      onSend(message);
      setMessage("");
      if (textareaRef.current) {
        textareaRef.current.style.height = "auto";
      }
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="relative">
      <textarea
        ref={textareaRef}
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Ask about the model, game analysis, or your bet..."
        disabled={isLoading || disabled}
        className={cn(
          "w-full px-4 py-3 pr-12 rounded-xl",
          "bg-white/[0.03] border border-white/[0.06]",
          "text-white/90 placeholder:text-white/30",
          "resize-none overflow-hidden",
          "focus:outline-none focus:border-emerald-500/30 focus:bg-white/[0.05]",
          "transition-colors duration-200",
          "backdrop-blur-sm",
          "text-sm leading-relaxed",
          disabled && "opacity-50 cursor-not-allowed"
        )}
        rows={1}
        style={{ minHeight: "48px", maxHeight: "120px" }}
      />
      <motion.button
        type="submit"
        disabled={!message.trim() || isLoading || disabled}
        className={cn(
          "absolute right-2 bottom-2 w-8 h-8 rounded-lg",
          "flex items-center justify-center",
          "bg-emerald-500/10 hover:bg-emerald-500/20",
          "border border-emerald-500/20",
          "text-emerald-400",
          "transition-colors duration-200",
          "disabled:opacity-30 disabled:cursor-not-allowed"
        )}
        whileHover={{ scale: disabled ? 1 : 1.05 }}
        whileTap={{ scale: disabled ? 1 : 0.95 }}
      >
        {isLoading ? (
          <Loader2 size={14} className="animate-spin" />
        ) : (
          <Send size={14} />
        )}
      </motion.button>
    </form>
  );
}

