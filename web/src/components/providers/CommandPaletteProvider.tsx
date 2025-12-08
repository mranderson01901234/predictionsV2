"use client";

import { ReactNode } from "react";
import { CommandPalette, useCommandPalette } from "@/components/ui/command-palette";

interface CommandPaletteProviderProps {
    children: ReactNode;
}

export function CommandPaletteProvider({ children }: CommandPaletteProviderProps) {
    const { open, setOpen } = useCommandPalette();

    return (
        <>
            {children}
            <CommandPalette open={open} onOpenChange={setOpen} />
        </>
    );
}

export default CommandPaletteProvider;

