import itertools
import json

from dgu.graph import TextWorldGraph
from dataclasses import dataclass
from typing import List, Iterator, Dict, Any, Tuple, OrderedDict, Set
from torch.utils.data import Sampler, Dataset


class TemporalDataBatchSampler(Sampler[List[int]]):
    def __init__(self, batch_size: int, event_seq_lens: List[int]) -> None:
        self.batch_size = batch_size
        # Calculate how many batches can be sampled per event sequence
        # and use the sum as the length.
        self.len = sum(
            (e_seq + self.batch_size - 1) // self.batch_size for e_seq in event_seq_lens
        )
        self.event_seq_accum_lens = list(itertools.accumulate(event_seq_lens))

    def __iter__(self) -> Iterator[List[int]]:
        """
        Create sequential batches based on the event sequence lengths.
        If there are some left-over events in a sequence, return those
        as a shorter batch first before continuing with the next event
        sequence.
        """
        batch = []
        prev_accum_len = 0
        for accum_len in self.event_seq_accum_lens:
            for idx in range(prev_accum_len, accum_len):
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            if len(batch) > 0:
                yield batch
                batch = []
            prev_accum_len = accum_len

    def __len__(self) -> int:
        return self.len


class TWCmdGenDataset(Dataset):
    """
    TextWorld Command Generation temporal graph event dataset.

    Each data point contains the following information:
        {
            "game": "game name",
            "step": [walkthrough step, random step],
            "observation": "observation...",
            "previous_action": "previous action...",
            "event_seq": [graph event, ...],
            "node_labels": [node label, ...],
            "edge_labels": [edge label, ...],
        }

    Node and edge labels are for the graph AFTER the given event sequence.

    There are four event types: node addtion/deletion and edge addition/deletion.
    Each node event contains the following information:
        {
            "type": "node-{add,delete}",
            "node_id": id for node to be added/deleted,
            "timestamp": timestamp for the event,
        }
    Each edge event contains the following information:
        {
            "type": "edge-{add,delete}",
            "src_id": id for src node to be added/deleted,
            "dst_id": id for dst node to be added/deleted,
            "timestamp": timestamp for the event,
        }
    """

    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]

    def __len__(self) -> int:
        return len(self.data)

    @classmethod
    def from_cmd_gen_data(cls, path: str) -> "TWCmdGenDataset":
        with open(path, "r") as f:
            raw_data = json.load(f)
            graph_index = json.loads(raw_data["graph_index"])
        data: List[Dict[str, Any]] = []
        for example in raw_data["examples"]:
            data.append(
                {
                    "game": example["game"],
                    "step": example["step"],
                    "observation": example["observation"],
                    "previous_action": example["previous_action"],
                }
            )
        return cls(data)

    @staticmethod
    def transform_commands_to_events(
        cmds: List[str],
        step: List[int],
        graph: TextWorldGraph,
    ) -> List[Dict[str, Any]]:
        # timestamp is the sum of the walkthrough step and the random step
        timestamp = sum(step)
        events: List[Dict[str, Any]] = []
        for cmd in cmds:
            cmd_type, src, dst, rel = cmd.split(" , ")
            if cmd_type == "add":
                if graph.has_edge(src, dst):
                    # the edge already exists, continue
                    continue
                # the edge doesn't exist, so add it
                # if src or dst doesn't exit, add it first.
                if not graph.has_node(src):
                    graph.add_node(src)
                    events.append(
                        {
                            "type": "node-add",
                            "node_id": graph.get_node_id(src),
                            "timestamp": timestamp,
                        }
                    )
                if not graph.has_node(dst):
                    graph.add_node(dst)
                    events.append(
                        {
                            "type": "node-add",
                            "node_id": graph.get_node_id(dst),
                            "timestamp": timestamp,
                        }
                    )

                # add the edge add event
                graph.add_edge(src, dst, rel)
                events.append(
                    {
                        "type": "edge-add",
                        "src_id": graph.get_node_id(src),
                        "dst_id": graph.get_node_id(dst),
                        "timestamp": timestamp,
                    }
                )
            elif cmd_type == "delete":
                if not graph.has_edge(src, dst):
                    # the edge doesn't exist, continue
                    continue

                # delete the edge and add event
                graph.remove_edge(src, dst)
                src_id = graph.get_node_id(src)
                dst_id = graph.get_node_id(dst)
                events.append(
                    {
                        "type": "edge-delete",
                        "src_id": src_id,
                        "dst_id": dst_id,
                        "timestamp": timestamp,
                    }
                )
                # if there are no edges, delete the nodes
                if graph.in_degree(src_id) == 0 and graph.out_degree(src_id) == 0:
                    graph.remove_node(src)
                    events.append(
                        {
                            "type": "node-delete",
                            "node_id": src_id,
                            "timestamp": timestamp,
                        }
                    )
                if graph.in_degree(dst_id) == 0 and graph.out_degree(dst_id) == 0:
                    graph.remove_node(dst)
                    events.append(
                        {
                            "type": "node-delete",
                            "node_id": dst_id,
                            "timestamp": timestamp,
                        }
                    )
            else:
                raise ValueError(f"Unknown command {cmd}")

        return events
