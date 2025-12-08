export default function GamesLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    // Games page uses its own integrated layout
    // Sidebar is hidden via LayoutWrapper
    return <>{children}</>;
}
