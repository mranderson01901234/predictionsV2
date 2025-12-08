"use client";

import { usePathname } from "next/navigation";
import { Sidebar } from "./Sidebar";

interface LayoutWrapperProps {
    children: React.ReactNode;
}

// Routes that should not show the sidebar
const NO_SIDEBAR_ROUTES = ["/games"];

export function LayoutWrapper({ children }: LayoutWrapperProps) {
    const pathname = usePathname();

    // Check if current path should hide sidebar
    const shouldHideSidebar = NO_SIDEBAR_ROUTES.some((route) =>
        pathname.startsWith(route)
    );

    if (shouldHideSidebar) {
        return (
            <main className="min-h-screen">
                {children}
            </main>
        );
    }

    return (
        <>
            <Sidebar />
            <main className="lg:pl-64 min-h-screen transition-all duration-300">
                {children}
            </main>
        </>
    );
}

export default LayoutWrapper;
