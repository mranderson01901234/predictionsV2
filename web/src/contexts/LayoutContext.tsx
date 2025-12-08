'use client';

import { createContext, useContext, useState, useCallback, useEffect, ReactNode } from 'react';

interface LayoutContextType {
  chatOpen: boolean;
  setChatOpen: (open: boolean) => void;
  toggleChat: () => void;
}

const LayoutContext = createContext<LayoutContextType | null>(null);

export function LayoutProvider({ children }: { children: ReactNode }) {
  // Default to open, persist preference
  const [chatOpen, setChatOpenState] = useState(true);

  // Load preference from localStorage on mount
  useEffect(() => {
    const stored = localStorage.getItem('predictr-chat-open');
    if (stored !== null) {
      setChatOpenState(stored === 'true');
    }
  }, []);

  const setChatOpen = useCallback((open: boolean) => {
    setChatOpenState(open);
    localStorage.setItem('predictr-chat-open', String(open));
  }, []);

  const toggleChat = useCallback(() => {
    setChatOpen(!chatOpen);
  }, [chatOpen, setChatOpen]);

  return (
    <LayoutContext.Provider value={{ chatOpen, setChatOpen, toggleChat }}>
      {children}
    </LayoutContext.Provider>
  );
}

export function useLayout() {
  const context = useContext(LayoutContext);
  if (!context) {
    throw new Error('useLayout must be used within LayoutProvider');
  }
  return context;
}

