import { getGames, getFullGameDetails, GameDetail } from "@/lib/mock_data";
import { LiveDashboard } from "@/components/live";

export default async function GamesPage() {
    const games = await getGames();

    // Fetch full details for all games
    const detailsMap: Record<string, GameDetail> = {};

    await Promise.all(
        games.map(async (game) => {
            const details = await getFullGameDetails(game.game_id);
            detailsMap[game.game_id] = details;
        })
    );

    return <LiveDashboard games={games} initialDetails={detailsMap} />;
}
