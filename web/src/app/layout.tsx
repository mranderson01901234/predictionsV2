import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { LayoutWrapper } from "@/components/layout/LayoutWrapper";
import { CommandPaletteProvider } from "@/components/providers/CommandPaletteProvider";
import { LayoutProvider } from "@/contexts/LayoutContext";

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
        className={`${geistSans.variable} ${geistMono.variable} antialiased h-screen overflow-hidden`}
      >
        <LayoutProvider>
          <CommandPaletteProvider>
            <LayoutWrapper>
              {children}
            </LayoutWrapper>
          </CommandPaletteProvider>
        </LayoutProvider>
      </body>
    </html>
  );
}
