import torch
import tqdm

from torch.utils.data import DataLoader
from typing import List, Dict, Tuple, Any
from torch_geometric.data import Data, Batch

from dgu.nn.graph_updater import StaticLabelDiscreteGraphUpdater
from dgu.data import (
    TWCmdGenGraphEventFreeRunDataset,
    TWCmdGenGraphEventDataCollator,
    read_label_vocab_files,
)
from dgu.preprocessor import SpacyPreprocessor
from dgu.metrics.f1 import F1
from dgu.metrics.exact_match import ExactMatch
from dgu.graph import (
    batch_to_data_list,
    data_to_networkx,
    networkx_to_rdf,
    update_rdf_graph,
)


def main(
    data_filename: str,
    ckpt_filename: str,
    word_vocab_path: str,
    node_vocab_path: str,
    relation_vocab_path: str,
    batch_size: int,
    device: str,
) -> None:
    dataset = TWCmdGenGraphEventFreeRunDataset(data_filename, batch_size)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=1)
    preprocessor = SpacyPreprocessor.load_from_file(word_vocab_path)
    labels, label_id_map = read_label_vocab_files(node_vocab_path, relation_vocab_path)
    collator = TWCmdGenGraphEventDataCollator(preprocessor, label_id_map)

    lm = StaticLabelDiscreteGraphUpdater.load_from_checkpoint(
        ckpt_filename,
        word_vocab_path=word_vocab_path,
        node_vocab_path=node_vocab_path,
        relation_vocab_path=relation_vocab_path,
    )
    lm.eval()
    lm = lm.to(device)
    graph_f1 = F1()
    graph_em = ExactMatch()

    game_id_to_step_data_graph: Dict[int, Tuple[Dict[str, Any], Data]] = {}
    with tqdm.tqdm(total=len(dataset)) as pbar:
        for batch in dataloader:
            # finished games are the ones that were in game_id_to_graph, but are not
            # part of the new batch
            for finished_game_id in game_id_to_step_data_graph.keys() - {
                game_id for game_id, _ in batch
            }:
                step_data, graph = game_id_to_step_data_graph.pop(finished_game_id)
                generated_rdfs = networkx_to_rdf(data_to_networkx(graph, labels))
                groundtruth_rdfs = update_rdf_graph(
                    set(step_data["previous_graph_seen"]), step_data["target_commands"]
                )
                graph_f1.update([generated_rdfs], [groundtruth_rdfs])  # type: ignore
                graph_em.update([generated_rdfs], [groundtruth_rdfs])  # type: ignore
                pbar.update()
                pbar.set_postfix(
                    {
                        "graph_f1": graph_f1.compute().item(),
                        "graph_em": graph_em.compute().item(),
                    }
                )

            # new games are the ones that were not in game_id_to_graph, but are now
            # part of the new batch.
            # due to Python's dictionary ordering (insertion order), new games are
            # added always to the end.
            for game_id, step_data in batch:
                if game_id in game_id_to_step_data_graph:
                    _, graph = game_id_to_step_data_graph[game_id]
                    game_id_to_step_data_graph[game_id] = (step_data, graph)
                else:
                    game_id_to_step_data_graph[game_id] = (
                        step_data,
                        Data(
                            x=torch.empty(0, dtype=torch.long),
                            node_last_update=torch.empty(0, 2, dtype=torch.long),
                            edge_index=torch.empty(2, 0, dtype=torch.long),
                            edge_attr=torch.empty(0, dtype=torch.long),
                            edge_last_update=torch.empty(0, 2, dtype=torch.long),
                        ).to(device),
                    )

            # sanity check
            assert [game_id for game_id, _ in batch] == [
                game_id for game_id in game_id_to_step_data_graph
            ]

            # construct a batch
            batched_obs: List[str] = []
            batched_prev_actions: List[str] = []
            batched_timestamps: List[int] = []
            graph_list: List[Data] = []
            for game_id, (step_data, graph) in game_id_to_step_data_graph.items():
                batched_obs.append(step_data["observation"])
                batched_prev_actions.append(step_data["previous_action"])
                batched_timestamps.append(step_data["timestamp"])
                graph_list.append(graph)

            # greedy decode
            with torch.no_grad():
                results_list = lm.greedy_decode(
                    collator.collate_step_inputs(
                        batched_obs, batched_prev_actions, batched_timestamps
                    ).to(device),
                    Batch.from_data_list(graph_list),
                )

            # update graphs in game_id_to_step_data_graph
            for (game_id, (step_data, _)), updated_graph in zip(
                game_id_to_step_data_graph.items(),
                batch_to_data_list(
                    results_list[-1]["updated_batched_graph"], batch_size
                ),
            ):
                game_id_to_step_data_graph[game_id] = (step_data, updated_graph)
    print(f"Free Run Graph F1: {graph_f1.compute()}")
    print(f"Free Run Graph EM: {graph_em.compute()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_filename")
    parser.add_argument("ckpt_filename")
    parser.add_argument("--word-vocab-path", default="vocabs/word_vocab.txt")
    parser.add_argument("--node-vocab-path", default="vocabs/node_vocab.txt")
    parser.add_argument("--relation-vocab-path", default="vocabs/relation_vocab.txt")
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main(
        args.data_filename,
        args.ckpt_filename,
        args.word_vocab_path,
        args.node_vocab_path,
        args.relation_vocab_path,
        args.batch_size,
        args.device,
    )