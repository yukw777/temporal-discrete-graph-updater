import torch
import matplotlib.pyplot as plt
import seaborn as sb
import random

from typing import List
from torch.utils.data import DataLoader

from tdgu.nn.graph_updater import TemporalDiscreteGraphUpdater
from tdgu.data import (
    TWCmdGenGraphEventDataset,
    TWCmdGenGraphEventDataCollator,
    read_label_vocab_files,
)
from tdgu.preprocessor import SpacyPreprocessor
from tdgu.constants import EVENT_TYPE_ID_MAP, EVENT_TYPES
from tdgu.nn.utils import calculate_node_id_offsets


def main(
    data_filename: str,
    ckpt_filename: str,
    num_datapoints: int,
    word_vocab_path: str,
    node_vocab_path: str,
    relation_vocab_path: str,
    num_dataloader_worker: int,
    device: str,
) -> None:
    dataset = TWCmdGenGraphEventDataset(data_filename)
    preprocessor = SpacyPreprocessor.load_from_file(word_vocab_path)
    labels, label_id_map = read_label_vocab_files(node_vocab_path, relation_vocab_path)
    sampled_ids = random.sample(range(len(dataset)), num_datapoints)
    dataloader = DataLoader(  # type: ignore
        [dataset[i] for i in sampled_ids],  # type: ignore
        batch_size=num_datapoints,
        collate_fn=TWCmdGenGraphEventDataCollator(preprocessor, label_id_map),
        pin_memory=True,
        num_workers=num_dataloader_worker,
    )

    lm = TemporalDiscreteGraphUpdater.load_from_checkpoint(
        ckpt_filename,
        word_vocab_path=word_vocab_path,
        node_vocab_path=node_vocab_path,
        relation_vocab_path=relation_vocab_path,
    )
    lm.eval()
    lm = lm.to(device)
    for batch in dataloader:
        batch = batch.to(device)
        with torch.no_grad():
            results_list = lm.greedy_decode(
                batch.step_input, batch.initial_batched_graph
            )
        # self_attn_weights_list: List[torch.Tensor] = []
        obs_graph_attn_weights_list: List[torch.Tensor] = []
        decoded_event_type_ids_list: List[torch.Tensor] = []
        decoded_event_src_ids_list: List[torch.Tensor] = []
        decoded_event_dst_ids_list: List[torch.Tensor] = []
        decoded_event_label_ids_list: List[torch.Tensor] = []
        node_id_offsets_list: List[torch.Tensor] = []
        for results in results_list:
            decoded_event_type_ids_list.append(results["decoded_event_type_ids"])
            decoded_event_src_ids_list.append(results["decoded_event_src_ids"])
            decoded_event_dst_ids_list.append(results["decoded_event_dst_ids"])
            decoded_event_label_ids_list.append(results["decoded_event_label_ids"])
            obs_graph_attn_weights_list.append(
                torch.stack(results["obs_graph_attn_weights"]).mean(dim=0)
                # (batch, 1, obs_len)
            )
            node_id_offsets_list.append(
                calculate_node_id_offsets(
                    results["decoded_event_type_ids"].size(0),
                    results["updated_batched_graph"].batch,
                )
                # (batch)
            )
        batch_obs_graph_attn_weights = torch.cat(obs_graph_attn_weights_list, dim=1)
        # (batch, num_graph_events, obs_len)
        batch_decoded_event_type_ids = torch.stack(decoded_event_type_ids_list, dim=-1)
        # (batch, num_graph_events)
        batch_decoded_event_src_ids = torch.stack(decoded_event_src_ids_list, dim=-1)
        # (batch, num_graph_events)
        batch_decoded_event_dst_ids = torch.stack(decoded_event_dst_ids_list, dim=-1)
        # (batch, num_graph_events)
        batch_decoded_event_label_ids = torch.stack(
            decoded_event_label_ids_list, dim=-1
        )
        # (batch, num_graph_events)
        batch_node_id_offsets = torch.stack(node_id_offsets_list, dim=-1)

        for i, (
            obs_graph_attn_weights,
            decoded_event_type_ids,
            decoded_event_src_ids,
            decoded_event_dst_ids,
            decoded_event_label_ids,
            node_id_offsets,
            obs_word_ids,
            obs_len,
            decoded_event_seq_len,
        ) in enumerate(
            zip(
                batch_obs_graph_attn_weights,
                batch_decoded_event_type_ids,
                batch_decoded_event_src_ids,
                batch_decoded_event_dst_ids,
                batch_decoded_event_label_ids,
                batch_node_id_offsets,
                batch.step_input.obs_word_ids,
                batch.step_input.obs_mask.sum(dim=1),
                (batch_decoded_event_type_ids == EVENT_TYPE_ID_MAP["end"]).nonzero(
                    as_tuple=True
                )[1],
            )
        ):
            if obs_len > 0 and decoded_event_seq_len > 0:
                fig, ax = plt.subplots(figsize=(30, 9))
                ax.xaxis.tick_top()
                sb.heatmap(
                    obs_graph_attn_weights[:decoded_event_seq_len, :obs_len].cpu(),
                    cmap="Blues",
                    square=True,
                )
                plt.xticks(
                    torch.arange(obs_len) + 0.5,
                    lm.preprocessor.convert_ids_to_tokens(obs_word_ids[:obs_len]),
                )
                plt.yticks(
                    torch.arange(decoded_event_seq_len) + 0.5,
                    [
                        (
                            EVENT_TYPES[event_type],
                            labels[
                                results_list[j]["updated_batched_graph"].x[
                                    src_id + offset
                                ]
                            ],
                            labels[
                                results_list[j]["updated_batched_graph"].x[
                                    dst_id + offset
                                ]
                            ],
                            labels[label],
                        )
                        if event_type.item()
                        in {
                            EVENT_TYPE_ID_MAP["edge-add"],
                            EVENT_TYPE_ID_MAP["edge-delete"],
                        }
                        else (EVENT_TYPES[event_type], labels[label])
                        if event_type.item()
                        in {
                            EVENT_TYPE_ID_MAP["node-add"],
                            EVENT_TYPE_ID_MAP["node-delete"],
                        }
                        else (EVENT_TYPES[event_type],)
                        for j, (event_type, src_id, dst_id, label, offset) in enumerate(
                            zip(
                                decoded_event_type_ids[:decoded_event_seq_len],
                                decoded_event_src_ids[:decoded_event_seq_len],
                                decoded_event_dst_ids[:decoded_event_seq_len],
                                decoded_event_label_ids[:decoded_event_seq_len],
                                node_id_offsets[:decoded_event_seq_len],
                            )
                        )
                    ],
                    rotation="horizontal",
                )
                plt.savefig(f"{'-'.join(map(str, batch.ids[i]))}_obs_graph_attn.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_filename")
    parser.add_argument("ckpt_filename")
    parser.add_argument("--num-datapoints", default=1, type=int)
    parser.add_argument("--word-vocab-path", default="vocabs/word_vocab.txt")
    parser.add_argument("--node-vocab-path", default="vocabs/node_vocab.txt")
    parser.add_argument("--relation-vocab-path", default="vocabs/relation_vocab.txt")
    parser.add_argument("--num-dataloader-worker", default=0, type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main(
        args.data_filename,
        args.ckpt_filename,
        args.num_datapoints,
        args.word_vocab_path,
        args.node_vocab_path,
        args.relation_vocab_path,
        args.num_dataloader_worker,
        args.device,
    )
