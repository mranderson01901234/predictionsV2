"use client";

import { useState, useCallback, useRef, useEffect, useMemo } from "react";
import { GameDetail } from "@/lib/mock_data";
import { ChatContext } from "@/lib/ai/context-builder";
import { UserAdjustment, UserAdjustments } from "@/lib/ai/types";
import { detectAdjustmentIntent, magnitudeToValue } from "@/lib/ai/adjustment-detector";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  isStreaming?: boolean;
}

export interface UseAIChatReturn {
  messages: ChatMessage[];
  isLoading: boolean;
  error: string | null;
  sendMessage: (content: string) => Promise<void>;
  clearChat: () => void;
  retryLast: () => Promise<void>;
  adjustments: UserAdjustment[];
  resetAdjustments: () => void;
  addAdjustment: (adjustment: Omit<UserAdjustment, 'id' | 'createdAt'>) => void;
}

export function useAIChat(gameContext?: ChatContext | GameDetail): UseAIChatReturn {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [adjustments, setAdjustments] = useState<UserAdjustment[]>([]);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Generate a session ID for security tracking
  const sessionId = useMemo(() => `session-${Date.now()}-${Math.random().toString(36).slice(2)}`, []);

  // Get game ID from context
  const gameId = gameContext
    ? ('game' in gameContext && gameContext.game?.game_id)
      ? gameContext.game.game_id
      : (gameContext as GameDetail)?.game_id
    : undefined;

  // Clear error when game changes
  useEffect(() => {
    setError(null);
  }, [gameId]);

  // Process adjustments from message content
  const processAdjustmentIntent = useCallback((message: string): UserAdjustment | null => {
    const intent = detectAdjustmentIntent(message);

    if (intent.detected && intent.category !== 'reset' && intent.direction) {
      return {
        id: `adj-${Date.now()}`,
        category: intent.category!,
        direction: intent.direction,
        magnitude: magnitudeToValue(intent.magnitude),
        scope: intent.scope || 'game',
        gameId: gameId,
        createdAt: new Date(),
      };
    }

    // Handle reset
    if (intent.detected && intent.category === 'reset') {
      setAdjustments([]);
      return null;
    }

    return null;
  }, [gameId]);

  const sendMessage = useCallback(
    async (content: string) => {
      if (!content.trim() || isLoading) return;

      // Check for adjustment intent and update local state
      const newAdjustment = processAdjustmentIntent(content);
      if (newAdjustment) {
        setAdjustments(prev => [...prev.filter(a => a.category !== newAdjustment.category), newAdjustment]);
      }

      // Add user message optimistically
      const userMessage: ChatMessage = {
        id: `user-${Date.now()}`,
        role: "user",
        content: content.trim(),
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, userMessage]);
      setIsLoading(true);
      setError(null);

      // Cancel any ongoing request
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }

      abortControllerRef.current = new AbortController();

      try {
        // Prepare conversation history (last 10 messages)
        const conversationHistory = messages
          .slice(-10)
          .map((m) => ({
            role: m.role,
            content: m.content,
          }));

        // Prepare user adjustments
        const userAdjustments: UserAdjustments = {
          active: adjustments,
        };

        const response = await fetch("/api/chat", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            message: content.trim(),
            conversationHistory,
            gameContext: gameContext,
            userAdjustments,
            sessionId,
          }),
          signal: abortControllerRef.current.signal,
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({ error: "Unknown error" }));
          throw new Error(errorData.error || `HTTP ${response.status}`);
        }

        // Create assistant message for streaming
        const assistantMessageId = `assistant-${Date.now()}`;
        const assistantMessage: ChatMessage = {
          id: assistantMessageId,
          role: "assistant",
          content: "",
          timestamp: new Date(),
          isStreaming: true,
        };

        setMessages((prev) => [...prev, assistantMessage]);

        // Stream response
        const reader = response.body?.getReader();
        if (!reader) {
          throw new Error("No response body");
        }

        const decoder = new TextDecoder();
        let fullContent = "";
        let chunkCount = 0;

        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n').filter(line => line.trim());

            for (const line of lines) {
              if (line.startsWith("data: ")) {
                const data = line.slice(6);
                if (data === "[DONE]") {
                  // Log final content length to verify completeness
                  console.log(`[Frontend] Stream complete. Total chunks: ${chunkCount}, Final content length: ${fullContent.length}`);
                  console.log(`[Frontend] Final content preview: "${fullContent.substring(0, 100)}..."`);
                  // Ensure final state update with complete content
                  setMessages((prev) =>
                    prev.map((m) =>
                      m.id === assistantMessageId
                        ? { ...m, content: fullContent, isStreaming: false }
                        : m
                    )
                  );
                  setIsLoading(false);
                  return;
                }

                try {
                  const parsed = JSON.parse(data);
                  if (parsed.error) {
                    throw new Error(parsed.error);
                  }
                  if (parsed.content) {
                    chunkCount++;
                    fullContent += parsed.content;
                    // Log periodically to track progress
                    if (chunkCount % 50 === 0 || chunkCount <= 5) {
                      console.log(`[Frontend] Chunk ${chunkCount}: content length now ${fullContent.length}`);
                    }
                    // Update UI immediately - don't batch updates
                    setMessages((prev) =>
                      prev.map((m) =>
                        m.id === assistantMessageId
                          ? { ...m, content: fullContent }
                          : m
                      )
                    );
                  }
                } catch (e) {
                  // Skip malformed JSON
                  console.warn("[Frontend] Failed to parse chunk:", e);
                }
              }
            }
          }
        } finally {
          // Ensure final state update with complete content
          // This is a safety net in case [DONE] wasn't received properly
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMessageId
                ? { ...m, content: fullContent, isStreaming: false }
                : m
            )
          );
          setIsLoading(false);
          console.log(`[Frontend] Finally block. Final content length: ${fullContent.length}`);
        }
      } catch (err) {
        if (err instanceof Error && err.name === "AbortError") {
          // User cancelled, remove the assistant message if it was created
          setMessages((prev) => {
            const lastMessage = prev[prev.length - 1];
            if (lastMessage && lastMessage.role === "assistant" && lastMessage.isStreaming) {
              return prev.slice(0, -1);
            }
            return prev;
          });
          return;
        }

        const errorMessage = err instanceof Error ? err.message : "Failed to send message";
        setError(errorMessage);
        setIsLoading(false);

        // Update assistant message to show error (if it exists)
        setMessages((prev) => {
          const lastMessage = prev[prev.length - 1];
          if (lastMessage && lastMessage.role === "assistant" && lastMessage.isStreaming) {
            return prev.map((m, idx) =>
              idx === prev.length - 1
                ? { ...m, content: 'Sorry, I encountered an error. Try again?', isStreaming: false }
                : m
            );
          }
          return prev;
        });
      }
    },
    [isLoading, messages, gameContext, adjustments, sessionId, processAdjustmentIntent]
  );

  const retryLast = useCallback(async () => {
    // Find last user message and retry
    const lastUserMessage = [...messages].reverse().find(m => m.role === 'user');
    if (lastUserMessage) {
      // Remove last assistant message
      setMessages(prev => {
        const lastAssistantIndex = prev.findLastIndex(m => m.role === 'assistant');
        if (lastAssistantIndex > -1) {
          return prev.slice(0, lastAssistantIndex);
        }
        return prev;
      });
      // Resend
      await sendMessage(lastUserMessage.content);
    }
  }, [messages, sendMessage]);

  const clearChat = useCallback(() => {
    setMessages([]);
    setError(null);
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  }, []);

  const resetAdjustments = useCallback(() => {
    setAdjustments([]);
  }, []);

  const addAdjustment = useCallback((adjustment: Omit<UserAdjustment, 'id' | 'createdAt'>) => {
    const newAdjustment: UserAdjustment = {
      ...adjustment,
      id: `adj-${Date.now()}`,
      createdAt: new Date(),
    };
    setAdjustments(prev => [
      ...prev.filter(a => a.category !== newAdjustment.category),
      newAdjustment
    ]);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  return {
    messages,
    isLoading,
    error,
    sendMessage,
    clearChat,
    retryLast,
    adjustments,
    resetAdjustments,
    addAdjustment,
  };
}
