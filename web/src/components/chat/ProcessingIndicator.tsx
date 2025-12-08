"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

interface ProcessingIndicatorProps {
  className?: string;
}

export function ProcessingIndicator({ className }: ProcessingIndicatorProps) {
  const [dots, setDots] = useState(".");

  useEffect(() => {
    const interval = setInterval(() => {
      setDots((prev) => (prev.length >= 3 ? "." : prev + "."));
    }, 500);
    return () => clearInterval(interval);
  }, []);

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      className={cn("flex gap-3 mb-4", className)}
    >
      <div className="flex-shrink-0 w-7 h-7 flex items-center justify-center">
        {/* Spacer to align with assistant messages */}
      </div>
      <div className="flex-1 max-w-[85%]">
        <div className="text-base text-white/80">
          Processing{dots}
        </div>
      </div>
    </motion.div>
  );
}

