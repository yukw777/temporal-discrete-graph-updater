import textworld
import torch
import matplotlib.pyplot as plt
import networkx as nx
import json

from torch_geometric.data import Data, Batch
from dgu.graph import (
    batch_to_data_list,
    data_to_networkx,
    networkx_to_rdf,
    update_rdf_graph,
)
from typing import Set, Tuple, List
from pathlib import Path

from dgu.nn.graph_updater import StaticLabelDiscreteGraphUpdater
from dgu.data import TWCmdGenGraphEventStepInput
from dgu.constants import COMMANDS_TO_IGNORE, EVENT_TYPES


class Env:
    def __init__(self, game_file: str) -> None:
        game_file_path = Path(game_file)
        self.game_file_type = game_file_path.suffix
        if self.game_file_type == ".z8":
            env_infos = textworld.EnvInfos(admissible_commands=True)
            self.env = textworld.start(game_file, env_infos)
            self.env = textworld.envs.wrappers.Filter(self.env)
        elif self.game_file_type == ".jsonl":
            self.step_pointer = 0
            with open(game_file) as f:
                self.steps = []
                for line in f:
                    self.steps.append(json.loads(line))
        else:
            raise ValueError(f"Unknown game file type: {self.game_file_type}")

    def reset(self) -> Tuple[str, List[str]]:
        if self.game_file_type == ".z8":
            obs, infos = self.env.reset()
            return obs, infos["admissible_commands"]
        elif self.game_file_type == ".jsonl":
            self.step_pointer = 0
            return (
                self.steps[self.step_pointer]["observation"],
                [self.steps[self.step_pointer]["taken_action"]],
            )
        else:
            raise ValueError(f"Unknown game file type: {self.game_file_type}")

    def step(self, action: str) -> Tuple[str, bool, List[str]]:
        if self.game_file_type == ".z8":
            obs, _, done, infos = self.env.step(action)
            return obs, done, infos["admissible_commands"]
        elif self.game_file_type == ".jsonl":
            self.step_pointer += 1
            if self.step_pointer == len(self.steps) - 1:
                done = True
            else:
                done = False
            return (
                self.steps[self.step_pointer]["observation"],
                done,
                [self.steps[self.step_pointer]["taken_action"]],
            )
        else:
            raise ValueError(f"Unknown game file type: {self.game_file_type}")


def main(
    game_file: str,
    ckpt_filename: str,
    word_vocab_path: str,
    node_vocab_path: str,
    relation_vocab_path: str,
    device: str,
    verbose: bool,
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

    env = Env(game_file)
    obs, admissible_cmds = env.reset()
    action = "restart"
    graph = Data(
        x=torch.empty(0, dtype=torch.long),
        node_last_update=torch.empty(0, 2, dtype=torch.long),
        edge_index=torch.empty(2, 0, dtype=torch.long),
        edge_attr=torch.empty(0, dtype=torch.long),
        edge_last_update=torch.empty(0, 2, dtype=torch.long),
    ).to(device)
    rdf_graph: Set[str] = set()

    # turn on interactive mode
    plt.ion()

    done = False
    while not done:
        # print the observation
        print(obs)

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

        # verbose output
        if verbose:
            batch_graph_cmds, _ = lm.generate_batch_graph_triples_seq(
                [results["decoded_event_type_ids"] for results in results_list],
                [results["decoded_event_src_ids"] for results in results_list],
                [results["decoded_event_dst_ids"] for results in results_list],
                [results["decoded_event_label_ids"] for results in results_list],
                [results["updated_batched_graph"] for results in results_list],
            )
            graph_cmds = batch_graph_cmds[0]
            print()
            print("--------graph cmds---------")
            for cmd in graph_cmds:
                print(cmd)
            print("------graph events---------")
            for results in results_list:
                for event_type_id, event_src_id, event_dst_id, event_label_id in zip(
                    results["decoded_event_type_ids"],
                    results["decoded_event_src_ids"],
                    results["decoded_event_dst_ids"],
                    results["decoded_event_label_ids"],
                ):
                    event_type = EVENT_TYPES[event_type_id]
                    event_label = lm.labels[event_label_id]
                    if event_type == "node-add":
                        print((event_type, event_label))
                    elif event_type == "node-delete":
                        event_src_label = lm.labels[
                            results["updated_batched_graph"].x[event_src_id]
                        ]
                        print(
                            (
                                event_type,
                                (event_src_id.item(), event_src_label),
                                event_label,
                            )
                        )
                    else:
                        event_src_label = lm.labels[
                            results["updated_batched_graph"].x[event_src_id]
                        ]
                        event_dst_label = lm.labels[
                            results["updated_batched_graph"].x[event_dst_id]
                        ]
                        print(
                            (
                                event_type,
                                (event_src_id.item(), event_src_label),
                                (event_dst_id.item(), event_dst_label),
                                event_dst_label,
                                event_label,
                            )
                        )
            print("------graph event rdfs-----")
            for rdf in sorted(networkx_to_rdf(data_to_networkx(graph, lm.labels))):
                print(rdf)
            print("------graph cmd rdfs------")
            rdf_graph = update_rdf_graph(rdf_graph, graph_cmds)
            for rdf in sorted(rdf_graph):
                print(rdf)
            print("---------------------------")
            print()

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
        avail_actions = [a for a in admissible_cmds if a not in COMMANDS_TO_IGNORE]
        print("Available actions:")
        for i, a in enumerate(avail_actions):
            print(f"{i}. {a}")
        action = avail_actions[int(input(">> "))]
        obs, done, admissible_cmds = env.step(action)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("game_filename")
    parser.add_argument("ckpt_filename")
    parser.add_argument("--word-vocab-path", default="vocabs/word_vocab.txt")
    parser.add_argument("--node-vocab-path", default="vocabs/node_vocab.txt")
    parser.add_argument("--relation-vocab-path", default="vocabs/relation_vocab.txt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verbose", default=False, action="store_true")
    args = parser.parse_args()
    main(
        args.game_filename,
        args.ckpt_filename,
        args.word_vocab_path,
        args.node_vocab_path,
        args.relation_vocab_path,
        args.device,
        args.verbose,
    )
