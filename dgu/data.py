import math
import itertools
import json
import pytorch_lightning as pl
import torch
import pickle

from tqdm import tqdm
from typing import Deque, List, Iterator, Dict, Any, Optional, Tuple, Set
from collections import defaultdict, deque
from torch.utils.data import Sampler, Dataset, DataLoader, get_worker_info
from hydra.utils import to_absolute_path
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass, field

from dgu.graph import TextWorldGraph
from dgu.preprocessor import SpacyPreprocessor
from dgu.nn.utils import compute_masks_from_event_type_ids
from dgu.constants import EVENT_TYPE_ID_MAP


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
                "event_seq": [graph event, ...],
            },
            ...
        ]


    There are four event types: node addtion/deletion and edge addition/deletion.
    Each node event contains the following information:
        {
            "type": "node-{add,delete}",
            "node_id": id for node to be added/deleted,
            "timestamp": timestamp for the event,
            "label": label for node to be added/deleted,
        }
    Each edge event contains the following information:
        {
            "type": "edge-{add,delete}",
            "src_id": id for src node to be added/deleted,
            "dst_id": id for dst node to be added/deleted,
            "timestamp": timestamp for the event,
            "label": label for edge to be added/deleted,
        }
    """

    def __init__(self, path: str) -> None:
        with open(path, "r") as f:
            raw_data = json.load(f)
        graph = TextWorldGraph()
        self.data: List[List[Dict[str, Any]]] = []
        walkthrough_examples: List[Dict[str, Any]] = []
        random_examples: List[Dict[str, Any]] = []
        batch: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
        curr_walkthrough_step = -1
        curr_game = ""

        for example in tqdm(
            # add a bogus game at the end to take care of the last datapoints
            raw_data["examples"] + [{"game": "", "step": [0, 0]}],
            desc="processing examples",
        ):
            game = example["game"]
            if curr_game == "":
                # if it's the first or last (bogus) game, set it right away
                curr_game = game
            walkthrough_step, _ = example["step"]
            if curr_walkthrough_step != walkthrough_step:
                # a new walkthrough step has been taken
                # add the walkthrough examples so far
                for timestamp, w_example in enumerate(walkthrough_examples):
                    event_seq = list(
                        itertools.chain.from_iterable(
                            graph.process_triplet_cmd(
                                curr_game, curr_walkthrough_step, timestamp, cmd
                            )
                            for cmd in w_example["target_commands"]
                        )
                    )
                    batch[(curr_game, curr_walkthrough_step)].append(
                        {
                            "game": curr_game,
                            "walkthrough_step": curr_walkthrough_step,
                            "observation": w_example["observation"],
                            "previous_action": w_example["previous_action"],
                            "timestamp": timestamp,
                            "event_seq": event_seq,
                        }
                    )

                # add the random examples
                for timestamp, r_example in enumerate(random_examples):
                    event_seq = list(
                        itertools.chain.from_iterable(
                            graph.process_triplet_cmd(
                                curr_game,
                                curr_walkthrough_step,
                                len(walkthrough_examples) + timestamp,
                                cmd,
                            )
                            for cmd in r_example["target_commands"]
                        )
                    )
                    batch[(curr_game, curr_walkthrough_step)].append(
                        {
                            "game": curr_game,
                            "walkthrough_step": curr_walkthrough_step,
                            "observation": r_example["observation"],
                            "previous_action": r_example["previous_action"],
                            "timestamp": len(walkthrough_examples) + timestamp,
                            "event_seq": event_seq,
                        }
                    )

                if curr_game != game:
                    # new game, so add the batch to the dataset
                    # and reset walkthrough_examples and batch
                    self.data.extend(batch.values())
                    walkthrough_examples = []
                    batch = defaultdict(list)
                    curr_game = game
                walkthrough_examples.append(example)
                random_examples = []
                curr_walkthrough_step = walkthrough_step
            else:
                # a new random step has been taken, so add to random_examples
                random_examples.append(example)

    def __getitem__(self, idx: int) -> List[Dict[str, Any]]:
        return self.data[idx]

    def __len__(self) -> int:
        return len(self.data)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, TWCmdGenTemporalDataset):
            return False
        return self.data == o.data


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
    node_ids: torch.Tensor = field(default_factory=empty_tensor)
    edge_ids: torch.Tensor = field(default_factory=empty_tensor)
    edge_index: torch.Tensor = field(default_factory=empty_tensor)
    edge_timestamps: torch.Tensor = field(default_factory=empty_tensor)
    tgt_event_timestamps: torch.Tensor = field(default_factory=empty_tensor)
    tgt_event_mask: torch.Tensor = field(default_factory=empty_tensor)
    tgt_event_type_ids: torch.Tensor = field(default_factory=empty_tensor)
    tgt_event_src_ids: torch.Tensor = field(default_factory=empty_tensor)
    tgt_event_src_mask: torch.Tensor = field(default_factory=empty_tensor)
    tgt_event_dst_ids: torch.Tensor = field(default_factory=empty_tensor)
    tgt_event_dst_mask: torch.Tensor = field(default_factory=empty_tensor)
    tgt_event_edge_ids: torch.Tensor = field(default_factory=empty_tensor)
    tgt_event_label_ids: torch.Tensor = field(default_factory=empty_tensor)
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
        return (
            self.node_ids.equal(o.node_ids)
            and self.edge_ids.equal(o.edge_ids)
            and self.edge_index.equal(o.edge_index)
            and self.edge_timestamps.equal(o.edge_timestamps)
            and self.tgt_event_timestamps.equal(o.tgt_event_timestamps)
            and self.tgt_event_mask.equal(o.tgt_event_mask)
            and self.tgt_event_type_ids.equal(o.tgt_event_type_ids)
            and self.tgt_event_src_ids.equal(o.tgt_event_src_ids)
            and self.tgt_event_src_mask.equal(o.tgt_event_src_mask)
            and self.tgt_event_dst_ids.equal(o.tgt_event_dst_ids)
            and self.tgt_event_dst_mask.equal(o.tgt_event_dst_mask)
            and self.tgt_event_edge_ids.equal(o.tgt_event_edge_ids)
            and self.tgt_event_label_ids.equal(o.tgt_event_label_ids)
            and self.groundtruth_event_type_ids.equal(o.groundtruth_event_type_ids)
            and self.groundtruth_event_src_ids.equal(o.groundtruth_event_src_ids)
            and self.groundtruth_event_src_mask.equal(o.groundtruth_event_src_mask)
            and self.groundtruth_event_dst_ids.equal(o.groundtruth_event_dst_ids)
            and self.groundtruth_event_dst_mask.equal(o.groundtruth_event_dst_mask)
            and self.groundtruth_event_label_ids.equal(o.groundtruth_event_label_ids)
            and self.groundtruth_event_mask.equal(o.groundtruth_event_mask)
        )

    def to(self, *args, **kwargs) -> "TWCmdGenTemporalGraphicalInput":
        return TWCmdGenTemporalGraphicalInput(
            node_ids=self.node_ids.to(*args, **kwargs),
            edge_ids=self.edge_ids.to(*args, **kwargs),
            edge_index=self.edge_index.to(*args, **kwargs),
            edge_timestamps=self.edge_timestamps.to(*args, **kwargs),
            tgt_event_timestamps=self.tgt_event_timestamps.to(*args, **kwargs),
            tgt_event_mask=self.tgt_event_mask.to(*args, **kwargs),
            tgt_event_type_ids=self.tgt_event_type_ids.to(*args, **kwargs),
            tgt_event_src_ids=self.tgt_event_src_ids.to(*args, **kwargs),
            tgt_event_src_mask=self.tgt_event_src_mask.to(*args, **kwargs),
            tgt_event_dst_ids=self.tgt_event_dst_ids.to(*args, **kwargs),
            tgt_event_dst_mask=self.tgt_event_dst_mask.to(*args, **kwargs),
            tgt_event_edge_ids=self.tgt_event_edge_ids.to(*args, **kwargs),
            tgt_event_label_ids=self.tgt_event_label_ids.to(*args, **kwargs),
            groundtruth_event_type_ids=self.groundtruth_event_type_ids.to(
                *args, **kwargs
            ),
            groundtruth_event_src_ids=self.groundtruth_event_src_ids.to(
                *args, **kwargs
            ),
            groundtruth_event_src_mask=self.groundtruth_event_src_mask.to(
                *args, **kwargs
            ),
            groundtruth_event_dst_ids=self.groundtruth_event_dst_ids.to(
                *args, **kwargs
            ),
            groundtruth_event_dst_mask=self.groundtruth_event_dst_mask.to(
                *args, **kwargs
            ),
            groundtruth_event_label_ids=self.groundtruth_event_label_ids.to(
                *args, **kwargs
            ),
            groundtruth_event_mask=self.groundtruth_event_mask.to(*args, **kwargs),
        )

    def pin_memory(self) -> "TWCmdGenTemporalGraphicalInput":
        return TWCmdGenTemporalGraphicalInput(
            node_ids=self.node_ids.pin_memory(),
            edge_ids=self.edge_ids.pin_memory(),
            edge_index=self.edge_index.pin_memory(),
            edge_timestamps=self.edge_timestamps.pin_memory(),
            tgt_event_timestamps=self.tgt_event_timestamps.pin_memory(),
            tgt_event_mask=self.tgt_event_mask.pin_memory(),
            tgt_event_type_ids=self.tgt_event_type_ids.pin_memory(),
            tgt_event_src_ids=self.tgt_event_src_ids.pin_memory(),
            tgt_event_src_mask=self.tgt_event_src_mask.pin_memory(),
            tgt_event_dst_ids=self.tgt_event_dst_ids.pin_memory(),
            tgt_event_dst_mask=self.tgt_event_dst_mask.pin_memory(),
            tgt_event_edge_ids=self.tgt_event_edge_ids.pin_memory(),
            tgt_event_label_ids=self.tgt_event_label_ids.pin_memory(),
            groundtruth_event_type_ids=self.groundtruth_event_type_ids.pin_memory(),
            groundtruth_event_src_ids=self.groundtruth_event_src_ids.pin_memory(),
            groundtruth_event_src_mask=self.groundtruth_event_src_mask.pin_memory(),
            groundtruth_event_dst_ids=self.groundtruth_event_dst_ids.pin_memory(),
            groundtruth_event_dst_mask=self.groundtruth_event_dst_mask.pin_memory(),
            groundtruth_event_label_ids=self.groundtruth_event_label_ids.pin_memory(),
            groundtruth_event_mask=self.groundtruth_event_mask.pin_memory(),
        )


@dataclass(frozen=True)
class TWCmdGenTemporalBatch:
    data: Tuple[
        Tuple[TWCmdGenTemporalTextualInput, Tuple[TWCmdGenTemporalGraphicalInput, ...]],
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
                )
                for textual, graphicals in self.data
            )
        )

    def pin_memory(self) -> "TWCmdGenTemporalBatch":
        return TWCmdGenTemporalBatch(
            data=tuple(
                (
                    textual.pin_memory(),
                    tuple(graphical.pin_memory() for graphical in graphicals),
                )
                for textual, graphicals in self.data
            )
        )

    def split(self, split_size: int) -> List["TWCmdGenTemporalBatch"]:
        return [
            TWCmdGenTemporalBatch(
                data=self.data[ndx : min(ndx + split_size, len(self))]
            )
            for ndx in range(0, len(self), split_size)
        ]


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
        self.worker_id_space_initialized = False

        self.preprocessor = preprocessor
        self.label_id_map = label_id_map

    @staticmethod
    def create_node_id_set() -> Set[int]:
        # always include the placeholder node 0
        return {0}

    @staticmethod
    def create_edges_dict() -> Dict[int, Tuple[int, int]]:
        # always include the placeholder, self-loop edge 0
        return {0: (0, 0)}

    @staticmethod
    def create_global_worker_id_tuple() -> Tuple[Dict[int, int], Dict[int, int]]:
        # always map worker node/edge ID 0 to the padding node/edge ID
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
            self.global_node_ids[(game, walkthrough_step)].add(event["node_id"])
        elif event_type == "edge-add":
            self.global_edges[(game, walkthrough_step)][event["edge_id"]] = (
                event["src_id"],
                event["dst_id"],
            )

    def init_worker_id_space(self, worker_info: Optional[Tuple[int, int]]) -> None:
        """
        Initialize the node/edge ID space for the worker. If info is None, there
        are no workers so assign the whole ID space. Otherwise, divide up the overall
        ID space into equal chunks and assign.

        worker_info: (worker_id, num_workers)
        """
        # NOTE: the ID space starts from 1 since 0 is for padding
        if worker_info is None:
            # main process, assign the whole ID space
            node_id_start = 1
            node_id_end = self.max_node_id
            edge_id_start = 1
            edge_id_end = self.max_edge_id
        else:
            # worker process, assign a chunk of the whole ID space
            worker_id, num_workers = worker_info

            def calc_start_end(
                min_id: int, max_id: int, worker_id: int, num_workers: int
            ) -> Tuple[int, int]:
                per_worker = int(math.ceil((max_id - min_id) / float(num_workers)))
                id_start = min_id + worker_id * per_worker
                id_end = min(id_start + per_worker, max_id)
                return id_start, id_end

            node_id_start, node_id_end = calc_start_end(
                1, self.max_node_id, worker_id, num_workers
            )
            edge_id_start, edge_id_end = calc_start_end(
                1, self.max_edge_id, worker_id, num_workers
            )
        self.unused_worker_node_ids: Deque[int] = deque(
            i for i in range(node_id_start, node_id_end)
        )
        self.unused_worker_edge_ids: Deque[int] = deque(
            i for i in range(edge_id_start, edge_id_end)
        )

        # {(game, walkthrough_step): (
        #       {global_node_id: worker_node_id, ...},
        #       {global_edge_id: worker_edge_id, ...})}
        self.allocated_global_worker_id_map: Dict[
            Tuple[str, int], Tuple[Dict[int, int], Dict[int, int]]
        ] = defaultdict(self.create_global_worker_id_tuple)

        self.global_node_ids: Dict[Tuple[str, int], Set[int]] = defaultdict(
            self.create_node_id_set
        )
        self.global_edges: Dict[
            Tuple[str, int], Dict[int, Tuple[int, int]]
        ] = defaultdict(self.create_edges_dict)

        # set the flag to True so it doesn't get reinitialized
        self.worker_id_space_initialized = True

    def allocate_worker_ids(self, game: str, walkthrough_step: int) -> None:
        """
        Allocate worker IDs to global IDs for the game.
        """
        # retrieve the global-to-worker node/edge ID map that needs to be updated
        (
            allocated_node_id_map,
            allocated_edge_id_map,
        ) = self.allocated_global_worker_id_map[(game, walkthrough_step)]

        # retrieve the global node IDs that have worker node IDs
        global_node_ids = self.global_node_ids[(game, walkthrough_step)]

        # assign worker node IDs to unallocated global node IDs
        for unallocated_global_node_id in (
            global_node_ids - allocated_node_id_map.keys()
        ):
            allocated_node_id_map[
                unallocated_global_node_id
            ] = self.unused_worker_node_ids.popleft()

        # retrieve the global edge IDs and edge indices that have worker node/edge IDs
        global_edges = self.global_edges[(game, walkthrough_step)]

        # assign worker edge IDs and node IDs to global edge IDs and edge indices
        for unallocated_global_edge_id in (
            global_edges.keys() - allocated_edge_id_map.keys()
        ):
            allocated_edge_id_map[
                unallocated_global_edge_id
            ] = self.unused_worker_edge_ids.popleft()

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
            "tgt_event_timestamps": (batch, event_seq_len),
            "tgt_event_mask": (batch, event_seq_len),
            "tgt_event_src_mask": (batch, event_seq_len),
            "tgt_event_dst_mask": (batch, event_seq_len),
            "tgt_event_label_ids": (batch, event_seq_len),
            "groundtruth_event_type_ids": (batch, event_seq_len),
            "groundtruth_event_mask": (batch, event_seq_len),
            "groundtruth_event_src_mask": (batch, event_seq_len),
            "groundtruth_event_dst_mask": (batch, event_seq_len),
            "groundtruth_event_label_ids": (batch, event_seq_len),
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
            tgt_event_mask,
            tgt_event_src_mask,
            tgt_event_dst_mask,
        ) = compute_masks_from_event_type_ids(tgt_event_type_ids)

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
            "tgt_event_timestamps": tgt_event_timestamps,
            "tgt_event_mask": tgt_event_mask,
            "tgt_event_src_mask": tgt_event_src_mask,
            "tgt_event_dst_mask": tgt_event_dst_mask,
            "tgt_event_label_ids": tgt_event_label_ids,
            "groundtruth_event_type_ids": groundtruth_event_type_ids,
            "groundtruth_event_mask": groundtruth_event_mask,
            "groundtruth_event_src_mask": groundtruth_event_src_mask,
            "groundtruth_event_dst_mask": groundtruth_event_dst_mask,
            "groundtruth_event_label_ids": groundtruth_event_label_ids,
        }

    def collate_graphical_inputs(
        self, batch_step: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Collate the graphical inputs of the given batch.

        output: {
            "node_ids": len({0: (batch, num_nodes), ...}) = event_seq_len,
            "edge_ids": len({0: (batch, num_edges), ...}) = event_seq_len,
            "edge_index": len({0: (batch, 2, num_edges), ...}) = event_seq_len,
            "edge_timestamps": len({0: (batch, num_edges), ...}) = event_seq_len,
            "tgt_event_src_ids": (batch, event_seq_len),
            "tgt_event_dst_ids": (batch, event_seq_len),
            "tgt_event_edge_ids": (batch, event_seq_len),
            "groundtruth_event_src_ids": (batch, event_seq_len),
            "groundtruth_event_dst_ids": (batch, event_seq_len),
        }
        """
        batch_event_src_ids: List[List[int]] = []
        batch_event_dst_ids: List[List[int]] = []
        batch_event_src_pos: List[List[int]] = []
        batch_event_dst_pos: List[List[int]] = []
        batch_event_edge_ids: List[List[int]] = []
        event_seq_node_ids: Dict[int, List[List[int]]] = defaultdict(list)
        event_seq_edge_ids: Dict[int, List[List[int]]] = defaultdict(list)
        event_seq_edge_index: Dict[int, List[List[Tuple[int, int]]]] = defaultdict(list)
        event_seq_edge_timestamps: Dict[int, List[List[float]]] = defaultdict(list)
        max_event_seq_len = max(
            len(step["event_seq"]) if step != {} else 0 for step in batch_step
        )
        for step in batch_step:
            event_src_ids: List[int] = []
            event_dst_ids: List[int] = []
            event_src_pos: List[int] = []
            event_dst_pos: List[int] = []
            event_edge_ids: List[int] = []
            if max_event_seq_len == 0:
                # no event sequence in this batch_step
                event_seq_node_ids[-1].append([0])
                event_seq_edge_ids[-1].append([0])
                event_seq_edge_index[-1].append([(0, 0)])
                event_seq_edge_timestamps[-1].append([0.0])
            for i in range(max_event_seq_len):
                if step == {} or i >= len(step["event_seq"]):
                    # we've passed the last event, so just add padded lists and move on
                    event_seq_node_ids[i].append([0])
                    event_seq_edge_ids[i].append([0])
                    event_seq_edge_index[i].append([(0, 0)])
                    event_seq_edge_timestamps[i].append([0.0])
                    if i == 0:
                        event_seq_node_ids[-1].append([0])
                        event_seq_edge_ids[-1].append([0])
                        event_seq_edge_index[-1].append([(0, 0)])
                        event_seq_edge_timestamps[-1].append([0.0])
                    continue
                game = step["game"]
                walkthrough_step = step["walkthrough_step"]
                event = step["event_seq"][i]
                if i == 0:
                    # add the previous subgraph node/edge IDs
                    # this is to support the initial event generation.
                    node_id_map, edge_id_map = self.allocated_global_worker_id_map[
                        (game, walkthrough_step)
                    ]
                    event_seq_node_ids[-1].append(
                        [
                            node_id_map[nid]
                            for nid in sorted(
                                self.global_node_ids[(game, walkthrough_step)]
                            )
                        ]
                    )
                    edge_id_list: List[int] = []
                    edge_index_list: List[Tuple[int, int]] = []
                    for edge_id, edge_index in self.global_edges[
                        (game, walkthrough_step)
                    ].items():
                        edge_id_list.append(edge_id)
                        edge_index_list.append(edge_index)
                    event_seq_edge_ids[-1].append(
                        [edge_id_map[eid] for eid in edge_id_list]
                    )
                    event_seq_edge_index[-1].append(
                        [
                            (node_id_map[src], node_id_map[dst])
                            for src, dst in edge_index_list
                        ]
                    )
                    event_seq_edge_timestamps[-1].append(
                        [float(event["timestamp"])] * len(edge_id_list)
                    )

                # update the graph with the graph events in the batch
                self.update_subgraph(game, walkthrough_step, event)

                # get the worker node/edge ID maps based on the updated graph
                self.allocate_worker_ids(game, walkthrough_step)
                node_id_map, edge_id_map = self.allocated_global_worker_id_map[
                    (game, walkthrough_step)
                ]

                # collect all the allocated worker node IDs
                event_node_ids = sorted(self.global_node_ids[(game, walkthrough_step)])
                event_seq_node_ids[i].append(
                    [node_id_map[nid] for nid in event_node_ids]
                )

                # create a map from global node ID => position in event_node_ids
                # which will be used for edge indices and groundtruth node IDs
                # this is necessary to calculate cross entropy losses
                node_id_pos_map = {
                    node_id: i for i, node_id in enumerate(event_node_ids)
                }

                edge_id_list = []
                edge_index_list = []
                for edge_id, edge_index in self.global_edges[
                    (game, walkthrough_step)
                ].items():
                    edge_id_list.append(edge_id)
                    edge_index_list.append(edge_index)
                event_seq_edge_ids[i].append([edge_id_map[eid] for eid in edge_id_list])
                event_seq_edge_index[i].append(
                    [
                        (node_id_map[src], node_id_map[dst])
                        for src, dst in edge_index_list
                    ]
                )
                event_seq_edge_timestamps[i].append(
                    [float(event["timestamp"])] * len(edge_id_list)
                )

                # collect event worker node/edge IDs
                if event["type"] in {"node-add", "node-delete"}:
                    event_src_ids.append(node_id_map[event["node_id"]])
                    # 0 if node-add as the node hasn't been added and does not have
                    # the correct position. Ultimately it's going to be masked.
                    if event["type"] == "node-add":
                        event_src_pos.append(0)
                    else:
                        event_src_pos.append(node_id_pos_map[event["node_id"]])
                    # used the placeholder node and edge
                    event_dst_ids.append(0)
                    event_dst_pos.append(0)
                    event_edge_ids.append(0)
                else:
                    event_src_ids.append(node_id_map[event["src_id"]])
                    event_src_pos.append(node_id_pos_map[event["src_id"]])
                    event_dst_ids.append(node_id_map[event["dst_id"]])
                    event_dst_pos.append(node_id_pos_map[event["dst_id"]])
                    event_edge_ids.append(edge_id_map[event["edge_id"]])
            batch_event_src_ids.append(event_src_ids)
            batch_event_src_pos.append(event_src_pos)
            batch_event_dst_ids.append(event_dst_ids)
            batch_event_dst_pos.append(event_dst_pos)
            batch_event_edge_ids.append(event_edge_ids)

        # placeholder node id for the start event
        tgt_event_src_ids = pad_sequence(
            [torch.tensor([0] + ids) for ids in batch_event_src_ids], batch_first=True
        )
        tgt_event_dst_ids = pad_sequence(
            [torch.tensor([0] + ids) for ids in batch_event_dst_ids], batch_first=True
        )
        # placeholder edge id for the start event
        tgt_event_edge_ids = pad_sequence(
            [torch.tensor([0] + ids) for ids in batch_event_edge_ids], batch_first=True
        )
        # placeholder node id for the end event
        groundtruth_event_src_ids = pad_sequence(
            [torch.tensor(pos + [0]) for pos in batch_event_src_pos], batch_first=True
        )
        groundtruth_event_dst_ids = pad_sequence(
            [torch.tensor(pos + [0]) for pos in batch_event_dst_pos], batch_first=True
        )

        node_ids: List[torch.Tensor] = [
            pad_sequence(
                [torch.tensor(step_nids) for step_nids in event_seq_node_ids[i]],
                batch_first=True,
            )
            for i in range(-1, max_event_seq_len)
        ]
        edge_ids: List[torch.Tensor] = [
            pad_sequence(
                [torch.tensor(step_eids) for step_eids in event_seq_edge_ids[i]],
                batch_first=True,
            )
            for i in range(-1, max_event_seq_len)
        ]
        edge_index_tensor: List[torch.Tensor] = [
            pad_sequence(
                [torch.tensor(step_eindex) for step_eindex in event_seq_edge_index[i]],
                batch_first=True,
            ).transpose(1, 2)
            for i in range(-1, max_event_seq_len)
        ]
        edge_timestamps: List[torch.Tensor] = [
            pad_sequence(
                [
                    torch.tensor(step_ets)
                    for step_ets in event_seq_edge_timestamps.get(
                        i, [[0.0] * len(batch_step)]
                    )
                ],
                batch_first=True,
            )
            for i in range(-1, max_event_seq_len)
        ]
        return {
            "node_ids": node_ids,
            "edge_ids": edge_ids,
            "edge_index": edge_index_tensor,
            "edge_timestamps": edge_timestamps,
            "tgt_event_src_ids": tgt_event_src_ids,
            "tgt_event_dst_ids": tgt_event_dst_ids,
            "tgt_event_edge_ids": tgt_event_edge_ids,
            "groundtruth_event_src_ids": groundtruth_event_src_ids,
            "groundtruth_event_dst_ids": groundtruth_event_dst_ids,
        }

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
                        edge_ids: (batch, num_edges),
                        edge_index: (batch, 2, num_edges),
                        edge_timestamps: (batch, num_edges),
                        tgt_event_timestamps: (batch, 1),
                        tgt_event_mask: (batch, 1),
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
                )
            ),
            ...
        )
        """
        # we initialize the worker ID space for every batch
        worker_info = get_worker_info()
        self.init_worker_id_space(
            (worker_info.id, worker_info.num_workers)
            if worker_info is not None
            else None
        )

        # start collating
        max_episode_len = max(len(episode) for episode in batch)
        collated_batch: List[
            Tuple[TWCmdGenTemporalTextualInput, List[TWCmdGenTemporalGraphicalInput]]
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

            graph_events: List[TWCmdGenTemporalGraphicalInput] = []
            non_graphical = self.collate_non_graphical_inputs(batch_ith_step)
            graphical = self.collate_graphical_inputs(batch_ith_step)
            for j in range(non_graphical["tgt_event_type_ids"].size(1)):
                graph_events.append(
                    TWCmdGenTemporalGraphicalInput(
                        node_ids=graphical["node_ids"][j],
                        edge_ids=graphical["edge_ids"][j],
                        edge_index=graphical["edge_index"][j],
                        edge_timestamps=graphical["edge_timestamps"][j],
                        tgt_event_timestamps=non_graphical["tgt_event_timestamps"][
                            :, j : j + 1
                        ],
                        tgt_event_mask=non_graphical["tgt_event_mask"][:, j : j + 1],
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
            collated_batch.append((textual, graph_events))
        return TWCmdGenTemporalBatch(
            data=tuple(
                (textual, tuple(graph_events))
                for textual, graph_events in collated_batch
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
        self.serialized_train_path = self.get_serialized_path(self.train_path)
        self.train_batch_size = train_batch_size
        self.train_num_worker = train_num_worker
        self.val_path = to_absolute_path(val_path)
        self.serialized_val_path = self.get_serialized_path(self.val_path)
        self.val_batch_size = val_batch_size
        self.val_num_worker = val_num_worker
        self.test_path = to_absolute_path(test_path)
        self.serialized_test_path = self.get_serialized_path(self.test_path)
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

    @staticmethod
    def get_serialized_path(raw_path: str) -> Path:
        path = Path(raw_path)
        return path.parent / (path.stem + ".pickle")

    @classmethod
    def serialize_dataset(cls, path: str, serialized_path: Path) -> None:
        if not serialized_path.exists():
            dataset = TWCmdGenTemporalDataset(path)
            with open(serialized_path, "wb") as f:
                pickle.dump(dataset, f)

    def prepare_data(self) -> None:
        self.serialize_dataset(self.train_path, self.serialized_train_path)
        self.serialize_dataset(self.val_path, self.serialized_val_path)
        self.serialize_dataset(self.test_path, self.serialized_test_path)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            with open(self.serialized_train_path, "rb") as f:
                self.train = pickle.load(f)
            with open(self.serialized_val_path, "rb") as f:
                self.valid = pickle.load(f)

        if stage == "test" or stage is None:
            with open(self.serialized_test_path, "rb") as f:
                self.test = pickle.load(f)

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
