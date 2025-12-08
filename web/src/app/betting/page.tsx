import { getGames, getAllGameDetails } from "@/lib/mock_data";
import { BettingCard } from "@/components/predictions/BettingCard";

export default async function BettingPage() {
    const games = await getGames();
    const details = await getAllGameDetails();

    return <BettingCard games={games} details={details} />;
}
