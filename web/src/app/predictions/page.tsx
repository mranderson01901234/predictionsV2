import { getGames, getAllGameDetails } from "@/lib/mock_data";
import { ModelProjections } from "@/components/predictions/ModelProjections";

export default async function PredictionsPage() {
    const games = await getGames();
    const details = await getAllGameDetails();

    return <ModelProjections games={games} details={details} />;
}
