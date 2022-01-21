import json


def main(train_filename: str, val_filename: str, test_filename: str) -> None:
    with open(train_filename) as f:
        raw_train_data = json.load(f)
    with open(val_filename) as f:
        raw_val_data = json.load(f)
    with open(test_filename) as f:
        raw_test_data = json.load(f)
    print(f'#Train: {len(raw_train_data["examples"])}')
    print(f'#Valid: {len(raw_val_data["examples"])}')
    print(f'#Test: {len(raw_test_data["examples"])}')
    num_total_examples = (
        len(raw_train_data["examples"])
        + len(raw_val_data["examples"])
        + len(raw_test_data["examples"])
    )

    obs_token_count = 0
    cmd_count = 0
    for example in (
        raw_train_data["examples"]
        + raw_val_data["examples"]
        + raw_test_data["examples"]
    ):
        obs_token_count += len(example["observation"].split())
        cmd_count += len(example["target_commands"])
    print(f"Avg. Obs.: {obs_token_count/num_total_examples}")
    print(f"Avg. #Operations: {cmd_count/num_total_examples}")

    edge_count = 0
    train_graph_index = json.loads(raw_train_data["graph_index"])
    for example in raw_train_data["examples"]:
        edge_count += len(
            train_graph_index["graphs"][str(example["previous_graph_seen"])]
        )
    print(f"Avg. #Connection: {edge_count/num_total_examples}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("train_filename")
    parser.add_argument("val_filename")
    parser.add_argument("test_filename")
    args = parser.parse_args()
    main(args.train_filename, args.val_filename, args.test_filename)
