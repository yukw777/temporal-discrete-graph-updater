from tdgu.data import TWCmdGenGraphEventDataset


def main(
    data_filename: str,
    game: str,
    walkthrough_step: int,
    random_step: int,
    graph_filename: str,
) -> None:
    dataset = TWCmdGenGraphEventDataset(data_filename)
    dataset.__getitem__(  # type: ignore
        dataset.idx_map.index((game, walkthrough_step, random_step)),
        draw_graph_filename=graph_filename,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_filename")
    parser.add_argument("game")
    parser.add_argument("walkthrough_step", type=int)
    parser.add_argument("random_step", type=int)
    parser.add_argument("graph_filename")
    args = parser.parse_args()
    main(
        args.data_filename,
        args.game,
        args.walkthrough_step,
        args.random_step,
        args.graph_filename,
    )
