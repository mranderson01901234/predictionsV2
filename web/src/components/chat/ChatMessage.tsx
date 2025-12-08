"use client";

import { ChatMessage as ChatMessageType } from "@/hooks/useAIChat";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { User, Brain } from "lucide-react";
import ReactMarkdown from "react-markdown";

interface ChatMessageProps {
  message: ChatMessageType;
  isStreaming?: boolean;
}

export function ChatMessage({ message, isStreaming }: ChatMessageProps) {
  const isUser = message.role === "user";

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn(
        "flex gap-3 mb-4",
        isUser ? "flex-row-reverse" : "flex-row"
      )}
    >
      {/* Avatar - only show for AI messages */}
      {!isUser && (
        <div className="flex-shrink-0 w-7 h-7 flex items-center justify-center">
          <Brain size={14} className="text-blue-400" />
        </div>
      )}

      {/* Message Content */}
      <div
        className="flex-1 max-w-[85%]"
      >
        <div
          className={cn(
            "leading-relaxed",
            isUser 
              ? "text-sm bg-white/15 backdrop-blur-sm rounded-xl px-4 py-3 border border-white/20 shadow-sm" 
              : "text-base text-white/80"
          )}
        >
          {isUser ? (
            <p className="text-white font-semibold">{message.content}</p>
          ) : (
            <>
              <ReactMarkdown
                components={{
                  // Only style bold text and paragraphs
                  p: ({ children }) => <p className="my-2 first:mt-0 last:mb-0 text-base">{children}</p>,
                  strong: ({ children }) => <strong className="font-bold text-white text-base">{children}</strong>,
                  // Render everything else as plain text
                  em: ({ children }) => <em className="text-base">{children}</em>,
                  code: ({ children }) => <span className="text-base">{children}</span>,
                  ul: ({ children }) => <ul className="list-disc ml-4 my-2 text-base">{children}</ul>,
                  ol: ({ children }) => <ol className="list-decimal ml-4 my-2 text-base">{children}</ol>,
                  li: ({ children }) => <li className="my-1 text-base">{children}</li>,
                  h1: ({ children }) => <p className="my-2 font-semibold text-lg">{children}</p>,
                  h2: ({ children }) => <p className="my-2 font-semibold text-lg">{children}</p>,
                  h3: ({ children }) => <p className="my-2 font-semibold text-base">{children}</p>,
                  blockquote: ({ children }) => <p className="my-2 text-base">{children}</p>,
                  a: ({ href, children }) => <span className="text-base">{children}</span>,
                  table: ({ children }) => <div className="my-2 text-base">{children}</div>,
                  th: ({ children }) => <span className="text-base">{children}</span>,
                  td: ({ children }) => <span className="text-base">{children}</span>,
                  hr: () => <br />,
                }}
              >
                {message.content}
              </ReactMarkdown>
            </>
          )}
        </div>
        <p className="text-[10px] text-white/30 mt-1.5">
          {message.timestamp.toLocaleTimeString([], {
            hour: "numeric",
            minute: "2-digit",
          })}
        </p>
      </div>
    </motion.div>
  );
}

