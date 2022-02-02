import json
import os

from tqdm import tqdm

from tdgu.data import TWCmdGenGraphEventDataset
from tdgu.constants import FOOD_COLORS


def main(data_filename: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    dataset = TWCmdGenGraphEventDataset(data_filename)
    with open(data_filename) as f:
        raw_data = json.load(f)

    for example in tqdm(raw_data["examples"]):
        for food, colors in FOOD_COLORS.items():
            same_food_count = 0
            for color in colors:
                if food + " " + color in example["observation"]:
                    same_food_count += 1
                if same_food_count > 1:
                    game = example["game"]
                    walkthrough_step, random_step = example["step"]
                    filename = (
                        game + f"walkthrough_step{walkthrough_step}"
                        f"+random_step{random_step}.jsonl"
                    )
                    with open(os.path.join(out_dir, filename), "w") as f:
                        walkthrough_examples = [
                            dataset.walkthrough_examples[(game, i)]
                            for i in range(walkthrough_step + 1)
                        ]
                        random_examples = dataset.random_examples[
                            (game, walkthrough_step)
                        ][:random_step]
                        game_steps = walkthrough_examples + random_examples
                        for i, game_step in enumerate(game_steps):
                            taken_action = (
                                "end"
                                if i == len(game_steps) - 1
                                else game_steps[i + 1]["previous_action"]
                            )
                            f.write(
                                json.dumps(
                                    {
                                        "observation": game_step["observation"],
                                        "taken_action": taken_action,
                                    }
                                )
                                + "\n"
                            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_filename")
    parser.add_argument("out_dir")
    args = parser.parse_args()
    main(args.data_filename, args.out_dir)
