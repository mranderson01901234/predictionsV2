import { getGames, getAllGameDetails } from "@/lib/mock_data";
import { ConfidencePool } from "@/components/predictions/ConfidencePool";

export default async function ConfidencePage() {
    const games = await getGames();
    const details = await getAllGameDetails();

    return <ConfidencePool games={games} details={details} />;
}
