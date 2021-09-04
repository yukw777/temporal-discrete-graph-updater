import itertools
import json
import pytorch_lightning as pl
import torch
import networkx as nx

from typing import Deque, List, Iterator, Dict, Any, Optional, Tuple, Set
from collections import defaultdict, deque
from torch.utils.data import Sampler, Dataset, DataLoader
from hydra.utils import to_absolute_path
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass, field
from torch_geometric.data import Data, Batch

from dgu.preprocessor import SpacyPreprocessor
from dgu.nn.utils import compute_masks_from_event_type_ids
from dgu.constants import EVENT_TYPE_ID_MAP
from dgu.graph import process_triplet_cmd


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


class TWCmdGenTemporalDataset(Dataset):
    """
    TextWorld Command Generation temporal graph event dataset.

    Each data point contains the following information:
        [
            {
                "game": "game name",
                "walkthrough_step": walkthrough step,
                "observation": "observation...",
                "previous_action": "previous action...",
                "timestamp": timestamp,
                "target_commands": [graph commands, ...],
            },
            ...
        ]
    """

    def __init__(self, path: str) -> None:
        with open(path, "r") as f:
            raw_data = json.load(f)
        self.walkthrough_examples: Dict[Tuple[str, int], Dict[str, Any]] = {}
        self.walkthrough_example_ids: List[Tuple[str, int]] = []
        self.random_examples: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(
            list
        )

        for example in raw_data["examples"]:
            game = example["game"]
            walkthrough_step, random_step = example["step"]
            if random_step == 0:
                # walkthrough example
                self.walkthrough_examples[(game, walkthrough_step)] = example
                self.walkthrough_example_ids.append((game, walkthrough_step))
            else:
                # random example
                self.random_examples[(game, walkthrough_step)].append(example)

    def __getitem__(self, idx: int) -> List[Dict[str, Any]]:
        game, walkthrough_step = self.walkthrough_example_ids[idx]
        walkthrough_examples = [
            self.walkthrough_examples[(game, i)] for i in range(walkthrough_step + 1)
        ]
        random_examples = self.random_examples[(game, walkthrough_step)]
        data: List[Dict[str, Any]] = []
        for timestamp, example in enumerate(walkthrough_examples + random_examples):
            data.append(
                {
                    "game": game,
                    "walkthrough_step": walkthrough_step,
                    "observation": example["observation"],
                    "previous_action": example["previous_action"],
                    "timestamp": timestamp,
                    "target_commands": example["target_commands"],
                }
            )
        return data

    def __len__(self) -> int:
        return len(self.walkthrough_example_ids)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, TWCmdGenTemporalDataset):
            return False
        return (
            self.walkthrough_examples == o.walkthrough_examples
            and self.walkthrough_example_ids == o.walkthrough_example_ids
            and self.random_examples == o.random_examples
        )


def empty_tensor() -> torch.Tensor:
    return torch.empty(0)


@dataclass(frozen=True)
class TWCmdGenTemporalTextualInput:
    obs_word_ids: torch.Tensor = field(default_factory=empty_tensor)
    obs_mask: torch.Tensor = field(default_factory=empty_tensor)
    prev_action_word_ids: torch.Tensor = field(default_factory=empty_tensor)
    prev_action_mask: torch.Tensor = field(default_factory=empty_tensor)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, TWCmdGenTemporalTextualInput):
            return False
        return (
            self.obs_word_ids.equal(o.obs_word_ids)
            and self.obs_mask.equal(o.obs_mask)
            and self.prev_action_word_ids.equal(o.prev_action_word_ids)
            and self.prev_action_mask.equal(o.prev_action_mask)
        )

    def to(self, *args, **kwargs) -> "TWCmdGenTemporalTextualInput":
        return TWCmdGenTemporalTextualInput(
            obs_word_ids=self.obs_word_ids.to(*args, **kwargs),
            obs_mask=self.obs_mask.to(*args, **kwargs),
            prev_action_word_ids=self.prev_action_word_ids.to(*args, **kwargs),
            prev_action_mask=self.prev_action_mask.to(*args, **kwargs),
        )

    def pin_memory(self) -> "TWCmdGenTemporalTextualInput":
        return TWCmdGenTemporalTextualInput(
            obs_word_ids=self.obs_word_ids.pin_memory(),
            obs_mask=self.obs_mask.pin_memory(),
            prev_action_word_ids=self.prev_action_word_ids.pin_memory(),
            prev_action_mask=self.prev_action_mask.pin_memory(),
        )


@dataclass(frozen=True)
class TWCmdGenTemporalGraphicalInput:
    node_label_ids: torch.Tensor = field(default_factory=empty_tensor)
    node_memory_update_index: torch.Tensor = field(default_factory=empty_tensor)
    node_memory_update_mask: torch.Tensor = field(default_factory=empty_tensor)
    edge_index: torch.Tensor = field(default_factory=empty_tensor)
    edge_label_ids: torch.Tensor = field(default_factory=empty_tensor)
    edge_last_update: torch.Tensor = field(default_factory=empty_tensor)
    edge_timestamps: torch.Tensor = field(default_factory=empty_tensor)
    batch: torch.Tensor = field(default_factory=empty_tensor)
    tgt_event_type_ids: torch.Tensor = field(default_factory=empty_tensor)
    tgt_event_src_ids: torch.Tensor = field(default_factory=empty_tensor)
    tgt_event_dst_ids: torch.Tensor = field(default_factory=empty_tensor)
    tgt_event_label_ids: torch.Tensor = field(default_factory=empty_tensor)
    tgt_event_timestamps: torch.Tensor = field(default_factory=empty_tensor)
    groundtruth_event_type_ids: torch.Tensor = field(default_factory=empty_tensor)
    groundtruth_event_src_ids: torch.Tensor = field(default_factory=empty_tensor)
    groundtruth_event_src_mask: torch.Tensor = field(default_factory=empty_tensor)
    groundtruth_event_dst_ids: torch.Tensor = field(default_factory=empty_tensor)
    groundtruth_event_dst_mask: torch.Tensor = field(default_factory=empty_tensor)
    groundtruth_event_label_ids: torch.Tensor = field(default_factory=empty_tensor)
    groundtruth_event_mask: torch.Tensor = field(default_factory=empty_tensor)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, TWCmdGenTemporalGraphicalInput):
            return False
        return all(getattr(self, f).equal(getattr(o, f)) for f in self.__annotations__)

    def to(self, *args, **kwargs) -> "TWCmdGenTemporalGraphicalInput":
        return TWCmdGenTemporalGraphicalInput(
            **{f: getattr(self, f).to(*args, **kwargs) for f in self.__annotations__}
        )

    def pin_memory(self) -> "TWCmdGenTemporalGraphicalInput":
        return TWCmdGenTemporalGraphicalInput(
            **{f: getattr(self, f).pin_memory() for f in self.__annotations__}
        )


@dataclass(frozen=True)
class TWCmdGenTemporalBatch:
    data: Tuple[
        Tuple[
            TWCmdGenTemporalTextualInput,
            Tuple[TWCmdGenTemporalGraphicalInput, ...],
            Tuple[Tuple[str, ...], ...],
        ],
        ...,
    ] = field(default_factory=tuple)

    def __len__(self) -> int:
        return len(self.data)

    def to(self, *args, **kwargs) -> "TWCmdGenTemporalBatch":
        return TWCmdGenTemporalBatch(
            data=tuple(
                (
                    textual.to(*args, **kwargs),
                    tuple(graphical.to(*args, **kwargs) for graphical in graphicals),
                    commands,
                )
                for textual, graphicals, commands in self.data
            )
        )

    def pin_memory(self) -> "TWCmdGenTemporalBatch":
        return TWCmdGenTemporalBatch(
            data=tuple(
                (
                    textual.pin_memory(),
                    tuple(graphical.pin_memory() for graphical in graphicals),
                    commands,
                )
                for textual, graphicals, commands in self.data
            )
        )

    def split(self, split_size: int) -> List["TWCmdGenTemporalBatch"]:
        return [
            TWCmdGenTemporalBatch(
                data=self.data[ndx : min(ndx + split_size, len(self))]
            )
            for ndx in range(0, len(self), split_size)
        ]


class TWCmdGenTemporalGraphData(Data):
    def __inc__(self, key: str, value: torch.Tensor) -> int:
        if key == "node_update_index":
            return self.node_memory_update_mask.sum()
        return super().__inc__(key, value)

    @classmethod
    def from_graph_event(
        cls,
        event_src_index: torch.Tensor,
        event_dst_index: torch.Tensor,
        timestamp: int,
        before_graph: nx.DiGraph,
        after_graph: nx.DiGraph,
        label_id_map: Dict[str, int],
    ) -> "TWCmdGenTemporalGraphData":
        node_id_map = {node: node_id for node_id, node in enumerate(after_graph.nodes)}
        node_memory_update_index: List[int] = []
        node_memory_update_mask: List[bool] = []
        for before_node in before_graph.nodes:
            if before_node in node_id_map:
                node_memory_update_index.append(node_id_map[before_node])
                node_memory_update_mask.append(True)
            else:
                node_memory_update_index.append(0)
                node_memory_update_mask.append(False)

        return TWCmdGenTemporalGraphData(
            # we set x for node_label_ids as that's how
            # Data figures out how many nodes there are
            x=torch.tensor([label_id_map[node.label] for node in after_graph.nodes])
            if after_graph.order() > 0
            else torch.empty(0, dtype=torch.long),
            node_memory_update_index=torch.tensor(node_memory_update_index)
            if node_memory_update_index
            else torch.empty(0, dtype=torch.long),
            node_memory_update_mask=torch.tensor(node_memory_update_mask)
            if node_memory_update_mask
            else torch.empty(0, dtype=torch.bool),
            edge_index=torch.tensor(
                [(node_id_map[src], node_id_map[dst]) for src, dst in after_graph.edges]
            ).t()
            if after_graph.number_of_edges() > 0
            else torch.empty(2, 0, dtype=torch.long),
            edge_attr=torch.tensor(
                [label_id_map[label] for _, _, label in after_graph.edges.data("label")]
            )
            if after_graph.number_of_edges() > 0
            else torch.empty(0, dtype=torch.long),
            edge_last_update=torch.tensor(
                [
                    last_update
                    for _, _, last_update in after_graph.edges.data("last_update")
                ],
                dtype=torch.float,
            )
            if after_graph.number_of_edges() > 0
            else torch.empty(0),
            edge_timestamps=torch.tensor(timestamp, dtype=torch.float).expand(
                after_graph.number_of_edges()
            ),
            event_src_index=event_src_index,
            event_dst_index=event_dst_index,
        )


class TWCmdGenTemporalDataCollator:
    def __init__(
        self,
        max_node_id: int,
        max_edge_id: int,
        preprocessor: SpacyPreprocessor,
        label_id_map: Dict[str, int],
    ) -> None:
        self.max_node_id = max_node_id
        self.max_edge_id = max_edge_id

        self.preprocessor = preprocessor
        self.label_id_map = label_id_map
        self.init_id_space()

    def init_id_space(self) -> None:
        self.unused_node_ids: Deque[int] = deque(i for i in range(1, self.max_node_id))
        self.unused_edge_ids: Deque[int] = deque(i for i in range(1, self.max_edge_id))

        # {(game, walkthrough_step): (
        #       {node_id: allocated_node_id, ...},
        #       {edge_id: allocated_edge_id, ...})}
        self.allocated_id_map: Dict[
            Tuple[str, int], Tuple[Dict[int, int], Dict[int, int]]
        ] = defaultdict(self.create_allocated_id_tuple)

        self.node_ids: Dict[Tuple[str, int], Set[int]] = defaultdict(
            self.create_node_id_set
        )
        self.node_labels: Dict[Tuple[str, int], Dict[int, str]] = defaultdict(
            self.create_node_label_dict
        )
        self.edges: Dict[Tuple[str, int], Dict[int, Tuple[int, int]]] = defaultdict(
            self.create_edges_dict
        )

    @staticmethod
    def create_node_id_set() -> Set[int]:
        # always include the placeholder node 0
        return {0}

    @staticmethod
    def create_node_label_dict() -> Dict[int, str]:
        # always include the placeholder node 0
        return {0: "<pad>"}

    @staticmethod
    def create_edges_dict() -> Dict[int, Tuple[int, int]]:
        # always include the placeholder, self-loop edge 0
        return {0: (0, 0)}

    @staticmethod
    def create_allocated_id_tuple() -> Tuple[Dict[int, int], Dict[int, int]]:
        # always map allocated node/edge ID 0 to the padding node/edge ID
        return ({0: 0}, {0: 0})

    def update_subgraph(
        self, game: str, walkthrough_step: int, event: Dict[str, Any]
    ) -> None:
        """
        Update subgraphs for games based on the given event.
        Basically keeps track of all the added nodes and edges.
        """
        event_type = event["type"]
        if event_type == "node-add":
            self.node_ids[(game, walkthrough_step)].add(event["node_id"])
            self.node_labels[(game, walkthrough_step)][event["node_id"]] = event[
                "label"
            ]
        elif event_type == "edge-add":
            self.edges[(game, walkthrough_step)][event["edge_id"]] = (
                event["src_id"],
                event["dst_id"],
            )

    def allocate_ids(self, game: str, walkthrough_step: int) -> None:
        """
        Allocate node/edge IDs for the game.
        """
        # retrieve the allocated node/edge ID map that needs to be updated
        allocated_node_id_map, allocated_edge_id_map = self.allocated_id_map[
            (game, walkthrough_step)
        ]

        # retrieve the node IDs that have allocated node IDs
        node_ids = self.node_ids[(game, walkthrough_step)]

        # assign node IDs to unallocated node IDs
        for unallocated_node_id in node_ids - allocated_node_id_map.keys():
            allocated_node_id_map[unallocated_node_id] = self.unused_node_ids.popleft()

        # retrieve the edge IDs and edge indices that have allocated node/edge IDs
        edges = self.edges[(game, walkthrough_step)]

        # assign edge IDs and node IDs to unallocated edge IDs and edge indices
        for unallocated_edge_id in edges.keys() - allocated_edge_id_map.keys():
            allocated_edge_id_map[unallocated_edge_id] = self.unused_edge_ids.popleft()

    def collate_textual_inputs(
        self, obs: List[str], prev_actions: List[str]
    ) -> TWCmdGenTemporalTextualInput:
        """
        Collate observations and previous actions.

        output: TWCmdGenTemporalTextualInput(
            obs_word_ids: (batch, obs_len),
            obs_mask: (batch, obs_len),
            prev_action_word_ids: (batch, prev_action_len),
            prev_action_mask: (batch, prev_action_len),
        )
        """
        # textual observation
        obs_word_ids, obs_mask = self.preprocessor.preprocess_tokenized(
            [ob.split() for ob in obs]
        )

        # textual previous action
        prev_action_word_ids, prev_action_mask = self.preprocessor.preprocess_tokenized(
            [prev_action.split() for prev_action in prev_actions]
        )
        return TWCmdGenTemporalTextualInput(
            obs_word_ids=obs_word_ids,
            obs_mask=obs_mask,
            prev_action_word_ids=prev_action_word_ids,
            prev_action_mask=prev_action_mask,
        )

    def collate_non_graphical_inputs(
        self, batch_step: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """
        Collate the non-graphical inputs of the given batch.

        output: {
            "tgt_event_type_ids": (batch, event_seq_len),
            "tgt_event_label_ids": (batch, event_seq_len),
            "tgt_event_timestamps": (batch, event_seq_len),
            "groundtruth_event_type_ids": (batch, event_seq_len),
            "groundtruth_event_src_mask": (batch, event_seq_len),
            "groundtruth_event_dst_mask": (batch, event_seq_len),
            "groundtruth_event_label_ids": (batch, event_seq_len),
            "groundtruth_event_mask": (batch, event_seq_len),
        }
        """
        # event types
        batch_event_type_ids = [
            [EVENT_TYPE_ID_MAP[event["type"]] for event in step["event_seq"]]
            if step != {}
            else []
            for step in batch_step
        ]
        # prepend a start event
        tgt_event_type_ids = pad_sequence(
            [
                torch.tensor([EVENT_TYPE_ID_MAP["start"]] + event_type_ids)
                for event_type_ids in batch_event_type_ids
            ],
            batch_first=True,
            padding_value=EVENT_TYPE_ID_MAP["pad"],
        )
        # append an end event
        groundtruth_event_type_ids = pad_sequence(
            [
                torch.tensor(event_type_ids + [EVENT_TYPE_ID_MAP["end"]])
                for event_type_ids in batch_event_type_ids
            ],
            batch_first=True,
            padding_value=EVENT_TYPE_ID_MAP["pad"],
        )

        (
            groundtruth_event_mask,
            groundtruth_event_src_mask,
            groundtruth_event_dst_mask,
        ) = compute_masks_from_event_type_ids(groundtruth_event_type_ids)

        # event timestamps
        batch_event_timestamps = [
            [event["timestamp"] for event in step["event_seq"]] if step != {} else []
            for step in batch_step
        ]
        tgt_event_timestamps = pad_sequence(
            [
                # 0 timestamp for start event
                torch.tensor([0.0] + event_timestamps)
                for event_timestamps in batch_event_timestamps
            ],
            batch_first=True,
        )

        # labels
        batch_event_label_ids = [
            [self.label_id_map[event["label"]] for event in step["event_seq"]]
            if step != {}
            else []
            for step in batch_step
        ]
        tgt_event_label_ids = pad_sequence(
            [
                torch.tensor([self.label_id_map[""]] + event_label_ids)
                for event_label_ids in batch_event_label_ids
            ],
            batch_first=True,
            padding_value=self.label_id_map[""],
        )
        groundtruth_event_label_ids = pad_sequence(
            [
                torch.tensor(event_label_ids + [self.label_id_map[""]])
                for event_label_ids in batch_event_label_ids
            ],
            batch_first=True,
            padding_value=self.label_id_map[""],
        )

        return {
            "tgt_event_type_ids": tgt_event_type_ids,
            "tgt_event_label_ids": tgt_event_label_ids,
            "tgt_event_timestamps": tgt_event_timestamps,
            "groundtruth_event_type_ids": groundtruth_event_type_ids,
            "groundtruth_event_src_mask": groundtruth_event_src_mask,
            "groundtruth_event_dst_mask": groundtruth_event_dst_mask,
            "groundtruth_event_label_ids": groundtruth_event_label_ids,
            "groundtruth_event_mask": groundtruth_event_mask,
        }

    def collate_graphical_inputs(
        self, batch_graphs: List[nx.DiGraph], batch_step: List[Dict[str, Any]]
    ) -> List[TWCmdGenTemporalGraphicalInput]:
        """
        Collate the graphical inputs of the given batch.

        output: len([TWCmdGenTemporalGraphicalInput(
            node_label_ids: (num_nodes),
            node_memory_update_index: (prev_num_nodes),
            node_memory_update_mask: (prev_num_nodes),
            edge_index: (2, num_edges),
            edge_label_ids: (num_edges),
            edge_last_update: (num_edges),
            edge_timestamps: (num_edges),
            batch: (num_nodes),
            tgt_event_type_ids: (batch),
            tgt_event_src_ids: (batch),
            tgt_event_dst_ids: (batch),
            tgt_event_label_ids: (batch),
            tgt_event_timestamps: (batch),
            groundtruth_event_type_ids: (batch),
            groundtruth_event_src_ids: (batch),
            groundtruth_event_src_mask: (batch),
            groundtruth_event_dst_ids: (batch),
            groundtruth_event_dst_mask: (batch),
            groundtruth_event_label_ids: (batch),
            groundtruth_event_mask: (batch),
        ), ...]) = event_seq_len
        """
        assert len(batch_graphs) == len(batch_step)
        batch_event_seq: List[List[Dict[str, Any]]] = []
        collated: List[TWCmdGenTemporalGraphicalInput] = []
        for graph, step in zip(batch_graphs, batch_step):
            if step == {}:
                batch_event_seq.append(
                    [
                        {
                            "type": "end",
                            "timestamp": 0,
                            "label": "",
                            # graphs don't change after this event
                            "before_graph": graph,
                            "after_graph": graph,
                        }
                    ]
                )
                continue
            curr_graph = graph
            event_seq: List[Dict[str, Any]] = []
            for cmd in step["target_commands"]:
                sub_event_seq = process_triplet_cmd(curr_graph, step["timestamp"], cmd)
                event_seq.extend(sub_event_seq)
                curr_graph = sub_event_seq[-1]["after_graph"]
            # add an end event
            event_seq += [
                {
                    "type": "end",
                    "timestamp": step["timestamp"],
                    "label": "",
                    # graphs don't change after this event
                    "before_graph": event_seq[-1]["after_graph"]
                    if event_seq
                    else nx.DiGraph(),
                    "after_graph": event_seq[-1]["after_graph"]
                    if event_seq
                    else nx.DiGraph(),
                }
            ]
            batch_event_seq.append(event_seq)

        max_event_seq_len = max(len(event_seq) for event_seq in batch_event_seq)
        batch_size = len(batch_step)

        # left shifted target events
        tgt_event_type_ids = torch.tensor([EVENT_TYPE_ID_MAP["start"]] * batch_size)
        tgt_event_src_ids = torch.tensor([0] * batch_size)
        tgt_event_dst_ids = torch.tensor([0] * batch_size)
        tgt_event_label_ids = torch.tensor([self.label_id_map[""]] * batch_size)
        tgt_event_timestamps = torch.tensor([0.0] * batch_size)
        for seq_step_num in range(max_event_seq_len):
            batch_event_type_ids: List[int] = []
            batch_event_src_ids: List[int] = []
            batch_event_dst_ids: List[int] = []
            batch_event_label_ids: List[int] = []
            batch_event_timestamps: List[int] = []
            batch_graph_data: List[TWCmdGenTemporalGraphData] = []
            for batch_item_num, event_seq in enumerate(batch_event_seq):
                # collect event data for all the items in the batch at event i
                if seq_step_num < len(event_seq):
                    event = event_seq[seq_step_num]
                    batch_event_type_ids.append(EVENT_TYPE_ID_MAP[event["type"]])
                    batch_event_src_ids.append(
                        event.get("src_id", event.get("node_id", 0))
                    )
                    batch_event_dst_ids.append(event.get("dst_id", 0))
                    batch_event_label_ids.append(self.label_id_map[event["label"]])
                    batch_event_timestamps.append(event["timestamp"])
                    batch_graph_data.append(
                        TWCmdGenTemporalGraphData.from_graph_event(
                            tgt_event_src_ids[batch_item_num],
                            tgt_event_dst_ids[batch_item_num],
                            event["timestamp"],
                            event["before_graph"],
                            event["after_graph"],
                            self.label_id_map,
                        )
                    )
                else:
                    batch_event_type_ids.append(EVENT_TYPE_ID_MAP["pad"])
                    batch_event_src_ids.append(0)
                    batch_event_dst_ids.append(0)
                    batch_event_label_ids.append(self.label_id_map[""])
                    batch_event_timestamps.append(0)
                    if seq_step_num == len(event_seq):
                        # we need to generate graph data to get rid of the sub graph
                        # for this batch item.
                        batch_graph_data.append(
                            TWCmdGenTemporalGraphData.from_graph_event(
                                torch.tensor(0),
                                torch.tensor(0),
                                0,
                                event_seq[-1]["after_graph"],
                                nx.DiGraph(),
                                self.label_id_map,
                            )
                        )
                    else:
                        batch_graph_data.append(
                            TWCmdGenTemporalGraphData(
                                x=torch.empty(0, dtype=torch.long),
                                node_memory_update_index=torch.empty(
                                    0, dtype=torch.long
                                ),
                                node_memory_update_mask=torch.empty(
                                    0, dtype=torch.bool
                                ),
                                edge_index=torch.empty(2, 0, dtype=torch.long),
                                edge_attr=torch.empty(0, dtype=torch.long),
                                edge_last_update=torch.empty(0),
                                edge_timestamps=torch.empty(0),
                                event_src_index=torch.tensor(0),
                                event_dst_index=torch.tensor(0),
                            )
                        )
            groundtruth_event_type_ids = torch.tensor(batch_event_type_ids)
            groundtruth_event_src_ids = torch.tensor(batch_event_src_ids)
            groundtruth_event_dst_ids = torch.tensor(batch_event_dst_ids)
            groundtruth_event_label_ids = torch.tensor(batch_event_label_ids)
            (
                groundtruth_event_mask,
                groundtruth_event_src_mask,
                groundtruth_event_dst_mask,
            ) = compute_masks_from_event_type_ids(groundtruth_event_type_ids)
            graph_batch = Batch.from_data_list(batch_graph_data)
            collated.append(
                TWCmdGenTemporalGraphicalInput(
                    node_label_ids=graph_batch.x,
                    node_memory_update_index=graph_batch.node_memory_update_index,
                    node_memory_update_mask=graph_batch.node_memory_update_mask,
                    edge_index=graph_batch.edge_index,
                    edge_label_ids=graph_batch.edge_attr,
                    edge_last_update=graph_batch.edge_last_update,
                    edge_timestamps=graph_batch.edge_timestamps,
                    batch=graph_batch.batch,
                    tgt_event_type_ids=tgt_event_type_ids,
                    tgt_event_src_ids=graph_batch.event_src_index,
                    tgt_event_dst_ids=graph_batch.event_dst_index,
                    tgt_event_label_ids=tgt_event_label_ids,
                    tgt_event_timestamps=tgt_event_timestamps,
                    groundtruth_event_type_ids=groundtruth_event_type_ids,
                    groundtruth_event_src_ids=groundtruth_event_src_ids,
                    groundtruth_event_src_mask=groundtruth_event_src_mask,
                    groundtruth_event_dst_ids=groundtruth_event_dst_ids,
                    groundtruth_event_dst_mask=groundtruth_event_dst_mask,
                    groundtruth_event_label_ids=groundtruth_event_label_ids,
                    groundtruth_event_mask=groundtruth_event_mask,
                )
            )

            # the current groundtruth events become the next target events
            tgt_event_type_ids = groundtruth_event_type_ids
            tgt_event_src_ids = groundtruth_event_src_ids
            tgt_event_dst_ids = groundtruth_event_dst_ids
            tgt_event_label_ids = groundtruth_event_label_ids
            tgt_event_timestamps = torch.tensor(
                batch_event_timestamps, dtype=torch.float
            )
        return collated

    def __call__(self, batch: List[List[Dict[str, Any]]]) -> TWCmdGenTemporalBatch:
        """
        Each element in the collated batch is a batched step. Each step
        has a dictionary of textual inputs and another dictionary
        of graph event inputs.
        (
            (
                TWCmdGenTemporalTextualInput(
                    obs_word_ids: (batch, obs_len),
                    obs_mask: (batch, obs_len),
                    prev_action_word_ids: (batch, prev_action_len),
                    prev_action_mask: (batch, prev_action_len),
                ),
                (
                    TWCmdGenTemporalGraphicalInput(
                        node_ids: (batch, num_nodes),
                        node_labels: [{node_id: node_label, ...}, ...],
                        edge_ids: (batch, num_edges),
                        edge_index: (batch, 2, num_edges),
                        edge_timestamps: (batch, num_edges),
                        tgt_event_timestamps: (batch, 1),
                        tgt_event_type_ids: (batch, 1),
                        tgt_event_src_ids: (batch, 1),
                        tgt_event_src_mask: (batch, 1),
                        tgt_event_dst_ids: (batch, 1),
                        tgt_event_dst_mask: (batch, 1),
                        tgt_event_edge_ids: (batch, 1),
                        tgt_event_label_ids: (batch, 1),
                        groundtruth_event_type_ids: (batch, 1),
                        groundtruth_event_src_ids: (batch, 1),
                        groundtruth_event_src_mask: (batch, 1),
                        groundtruth_event_dst_ids: (batch, 1),
                        groundtruth_event_dst_mask: (batch, 1),
                        groundtruth_event_label_ids: (batch, 1),
                        groundtruth_event_mask: (batch, 1),
                    ),
                    ...
                ),
                (
                    (commands, ...),
                    ...
                )
            ),
            ...
        )
        """
        self.init_id_space()
        max_episode_len = max(len(episode) for episode in batch)
        collated_batch: List[
            Tuple[
                TWCmdGenTemporalTextualInput,
                List[TWCmdGenTemporalGraphicalInput],
                Tuple[Tuple[str, ...], ...],
            ]
        ] = []
        for i in range(max_episode_len):
            batch_ith_step = [
                episode[i] if i < len(episode) else {} for episode in batch
            ]
            textual = self.collate_textual_inputs(
                # use "<bos> <eos>" for empty observations and previous actions
                # to prevent nan's when encoding.
                [step.get("observation", "<bos> <eos>") for step in batch_ith_step],
                [step.get("previous_action", "<bos> <eos>") for step in batch_ith_step],
            )
            cmds = tuple(tuple(step.get("commands", [])) for step in batch_ith_step)

            graph_events: List[TWCmdGenTemporalGraphicalInput] = []
            non_graphical = self.collate_non_graphical_inputs(batch_ith_step)
            graphical = self.collate_graphical_inputs(batch_ith_step)
            for j in range(non_graphical["tgt_event_type_ids"].size(1)):
                graph_events.append(
                    TWCmdGenTemporalGraphicalInput(
                        node_ids=graphical["node_ids"][j],
                        node_labels=graphical["node_labels"][j],
                        node_mask=graphical["node_masks"][j],
                        edge_ids=graphical["edge_ids"][j],
                        edge_index=graphical["edge_index"][j],
                        edge_timestamps=graphical["edge_timestamps"][j],
                        tgt_event_timestamps=non_graphical["tgt_event_timestamps"][
                            :, j : j + 1
                        ],
                        tgt_event_type_ids=non_graphical["tgt_event_type_ids"][
                            :, j : j + 1
                        ],
                        tgt_event_src_ids=graphical["tgt_event_src_ids"][:, j : j + 1],
                        tgt_event_src_mask=non_graphical["tgt_event_src_mask"][
                            :, j : j + 1
                        ],
                        tgt_event_dst_ids=graphical["tgt_event_dst_ids"][:, j : j + 1],
                        tgt_event_dst_mask=non_graphical["tgt_event_dst_mask"][
                            :, j : j + 1
                        ],
                        tgt_event_edge_ids=graphical["tgt_event_edge_ids"][
                            :, j : j + 1
                        ],
                        tgt_event_label_ids=non_graphical["tgt_event_label_ids"][
                            :, j : j + 1
                        ],
                        groundtruth_event_type_ids=non_graphical[
                            "groundtruth_event_type_ids"
                        ][:, j : j + 1],
                        groundtruth_event_src_ids=graphical[
                            "groundtruth_event_src_ids"
                        ][:, j : j + 1],
                        groundtruth_event_src_mask=non_graphical[
                            "groundtruth_event_src_mask"
                        ][:, j : j + 1],
                        groundtruth_event_dst_ids=graphical[
                            "groundtruth_event_dst_ids"
                        ][:, j : j + 1],
                        groundtruth_event_dst_mask=non_graphical[
                            "groundtruth_event_dst_mask"
                        ][:, j : j + 1],
                        groundtruth_event_label_ids=non_graphical[
                            "groundtruth_event_label_ids"
                        ][:, j : j + 1],
                        groundtruth_event_mask=non_graphical["groundtruth_event_mask"][
                            :, j : j + 1
                        ],
                    )
                )
            collated_batch.append((textual, graph_events, cmds))
        return TWCmdGenTemporalBatch(
            data=tuple(
                (textual, tuple(graph_events), tuple(cmds))
                for textual, graph_events, cmds in collated_batch
            )
        )


class TWCmdGenTemporalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_path: str,
        train_batch_size: int,
        train_num_worker: int,
        val_path: str,
        val_batch_size: int,
        val_num_worker: int,
        test_path: str,
        test_batch_size: int,
        test_num_worker: int,
        word_vocab_file: str,
        node_vocab_file: str,
        relation_vocab_file: str,
        max_num_nodes: int,
        max_num_edges: int,
    ) -> None:
        super().__init__()
        self.train_path = to_absolute_path(train_path)
        self.train_batch_size = train_batch_size
        self.train_num_worker = train_num_worker
        self.val_path = to_absolute_path(val_path)
        self.val_batch_size = val_batch_size
        self.val_num_worker = val_num_worker
        self.test_path = to_absolute_path(test_path)
        self.test_batch_size = test_batch_size
        self.test_num_worker = test_num_worker

        self.preprocessor = SpacyPreprocessor.load_from_file(
            to_absolute_path(word_vocab_file)
        )
        self.label_id_map = read_label_vocab_files(
            to_absolute_path(node_vocab_file), to_absolute_path(relation_vocab_file)
        )

        self.max_num_nodes = max_num_nodes
        self.max_num_edges = max_num_edges

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train = TWCmdGenTemporalDataset(self.train_path)
            self.valid = TWCmdGenTemporalDataset(self.val_path)

        if stage == "test" or stage is None:
            self.test = TWCmdGenTemporalDataset(self.test_path)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.train_batch_size,
            collate_fn=TWCmdGenTemporalDataCollator(
                self.max_num_nodes,
                self.max_num_edges,
                self.preprocessor,
                self.label_id_map,
            ),
            shuffle=True,
            pin_memory=True,
            num_workers=self.train_num_worker,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid,
            batch_size=self.val_batch_size,
            collate_fn=TWCmdGenTemporalDataCollator(
                self.max_num_nodes,
                self.max_num_edges,
                self.preprocessor,
                self.label_id_map,
            ),
            pin_memory=True,
            num_workers=self.val_num_worker,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            batch_size=self.val_batch_size,
            collate_fn=TWCmdGenTemporalDataCollator(
                self.max_num_nodes,
                self.max_num_edges,
                self.preprocessor,
                self.label_id_map,
            ),
            pin_memory=True,
            num_workers=self.test_num_worker,
        )


def read_label_vocab_files(
    node_vocab_file: str, relation_vocab_file: str
) -> Dict[str, int]:
    id_map: Dict[str, int] = {"": 0}
    with open(node_vocab_file) as f:
        for i, line in enumerate(f):
            stripped = line.strip()
            if stripped != "":
                id_map[stripped] = i + 1
    num_node_label = len(id_map)
    with open(relation_vocab_file) as f:
        for i, line in enumerate(f):
            stripped = line.strip()
            if stripped != "":
                id_map[stripped] = i + num_node_label
    return id_map
