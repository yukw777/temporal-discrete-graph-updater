import textworld
import torch
import matplotlib.pyplot as plt
import networkx as nx

from torch_geometric.data import Data, Batch
from dgu.graph import batch_to_data_list, data_to_networkx

from dgu.nn.graph_updater import StaticLabelDiscreteGraphUpdater
from dgu.data import TWCmdGenGraphEventStepInput
from dgu.constants import COMMANDS_TO_IGNORE


def main(
    game_file: str,
    ckpt_filename: str,
    word_vocab_path: str,
    node_vocab_path: str,
    relation_vocab_path: str,
    device: str,
) -> None:
    # load the model
    lm = StaticLabelDiscreteGraphUpdater.load_from_checkpoint(
        ckpt_filename,
        word_vocab_path=word_vocab_path,
        node_vocab_path=node_vocab_path,
        relation_vocab_path=relation_vocab_path,
    )
    lm.eval()
    lm = lm.to(device)

    env_infos = textworld.EnvInfos(admissible_commands=True)
    env = textworld.start(game_file, env_infos)
    env = textworld.envs.wrappers.Filter(env)
    obs, infos = env.reset()
    action = "restart"
    graph = Data(
        x=torch.empty(0, dtype=torch.long),
        node_last_update=torch.empty(0, 2, dtype=torch.long),
        edge_index=torch.empty(2, 0, dtype=torch.long),
        edge_attr=torch.empty(0, dtype=torch.long),
        edge_last_update=torch.empty(0, 2, dtype=torch.long),
    ).to(device)

    # turn on interactive mode
    plt.ion()

    done = False
    while not done:
        # print the observation
        print(obs)

        # print the available actions
        avail_actions = [
            a for a in infos["admissible_commands"] if a not in COMMANDS_TO_IGNORE
        ]
        print("Available actions:")
        for i, a in enumerate(avail_actions):
            print(f"{i}. {a}")

        # update the graph to the current step
        obs_word_ids, obs_mask = lm.preprocessor.clean_and_preprocess([obs])
        prev_action_word_ids, prev_action_mask = lm.preprocessor.clean_and_preprocess(
            [action]
        )
        with torch.no_grad():
            results_list = lm.greedy_decode(
                TWCmdGenGraphEventStepInput(
                    obs_word_ids=obs_word_ids,
                    obs_mask=obs_mask,
                    prev_action_word_ids=prev_action_word_ids,
                    prev_action_mask=prev_action_mask,
                    timestamps=torch.tensor([0]),
                ).to(device),
                Batch.from_data_list([graph]),
            )
        graph = batch_to_data_list(results_list[-1]["updated_batched_graph"], 1)[0]

        # visualize the graph
        nx_graph = data_to_networkx(graph, lm.labels)
        plt.figure(figsize=(30, 30))
        pos = nx.spring_layout(nx_graph)
        nx.draw(nx_graph, pos=pos, labels=nx.get_node_attributes(nx_graph, "label"))
        nx.draw_networkx_edge_labels(
            nx_graph, pos, edge_labels=nx.get_edge_attributes(nx_graph, "label")
        )
        plt.show()

        # get the next action
        action = avail_actions[int(input(">> "))]
        obs, _, done, infos = env.step(action)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("game_filename")
    parser.add_argument("ckpt_filename")
    parser.add_argument("--word-vocab-path", default="vocabs/word_vocab.txt")
    parser.add_argument("--node-vocab-path", default="vocabs/node_vocab.txt")
    parser.add_argument("--relation-vocab-path", default="vocabs/relation_vocab.txt")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main(
        args.game_filename,
        args.ckpt_filename,
        args.word_vocab_path,
        args.node_vocab_path,
        args.relation_vocab_path,
        args.device,
    )
