import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Sidebar } from "@/components/layout/Sidebar";
import { CommandPaletteProvider } from "@/components/providers/CommandPaletteProvider";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Predictr | Premium Sports Analytics",
  description: "Advanced sports predictions and market analytics.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased min-h-screen`}
      >
        <CommandPaletteProvider>
          <Sidebar />
          {/* Main content area - adjusts for sidebar */}
          <main className="lg:pl-64 min-h-screen transition-all duration-300">
            {children}
          </main>
        </CommandPaletteProvider>
      </body>
    </html>
  );
}
