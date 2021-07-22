import math
import itertools
import json
import pytorch_lightning as pl
import torch
import pickle

from tqdm import tqdm
from typing import Deque, List, Iterator, Dict, Any, Optional, Tuple, Set
from collections import defaultdict, deque
from torch.utils.data import Sampler, Dataset, DataLoader
from hydra.utils import to_absolute_path
from functools import partial
from pathlib import Path

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
    ) -> Dict[str, torch.Tensor]:
        """
        Collate observations and previous actions.

        output: {
            "obs_word_ids": (batch, obs_len),
            "obs_mask": (batch, obs_len),
            "prev_action_word_ids": (batch, prev_action_len),
            "prev_action_mask": (batch, prev_action_len),
        }
        """
        # textual observation
        obs_word_ids, obs_mask = self.preprocessor.preprocess_tokenized(
            [ob.split() for ob in obs]
        )

        # textual previous action
        prev_action_word_ids, prev_action_mask = self.preprocessor.preprocess_tokenized(
            [prev_action.split() for prev_action in prev_actions]
        )
        return {
            "obs_word_ids": obs_word_ids,
            "obs_mask": obs_mask,
            "prev_action_word_ids": prev_action_word_ids,
            "prev_action_mask": prev_action_mask,
        }

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
            [event["timestamp"] for event in step["event_seq"]] for step in batch_step
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

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass


class TWCmdGenTemporalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_path: str,
        train_batch_size: int,
        val_path: str,
        val_batch_size: int,
        test_path: str,
        test_batch_size: int,
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
        self.val_path = to_absolute_path(val_path)
        self.serialized_val_path = self.get_serialized_path(self.val_path)
        self.val_batch_size = val_batch_size
        self.test_path = to_absolute_path(test_path)
        self.serialized_test_path = self.get_serialized_path(self.test_path)
        self.test_batch_size = test_batch_size

        self.preprocessor = SpacyPreprocessor.load_from_file(
            to_absolute_path(word_vocab_file)
        )
        self.label_id_map = self.read_label_vocab_files(
            to_absolute_path(node_vocab_file), to_absolute_path(relation_vocab_file)
        )

        self.max_num_nodes = max_num_nodes
        self.max_num_edges = max_num_edges
        self.unused_node_ids: Deque[int] = deque(i for i in range(max_num_nodes))
        self.unused_edge_ids: Deque[int] = deque(i for i in range(max_num_edges))
        # {(game, walkthrough_step): (
        #       {global_node_id: local_node_id, ...},
        #       {global_edge_id: local_edge_id, ...})}
        self.used_ids: Dict[Tuple[str, int], Tuple[Dict[int, int], Dict[int, int]]] = {}

        self.global_node_ids: Dict[Tuple[str, int], Set[int]] = defaultdict(set)
        self.global_edges: Dict[
            Tuple[str, int], Dict[int, Tuple[int, int]]
        ] = defaultdict(dict)

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

    def allocate_local_ids(self, game: str, walkthrough_step: int) -> None:
        """
        Allocate local IDs to global IDs in the batch.
        """
        # get the difference between the (game, walkthrough_step)'s in the batch
        # and the ones currently in the memory.
        gw_set = set([(game, walkthrough_step)])
        old_gw_set = sorted(self.used_ids.keys() - gw_set)
        common_gw_set = sorted(gw_set.intersection(self.used_ids.keys()))
        new_gw_set = sorted(gw_set - self.used_ids.keys())

        # first free the node and edge IDs for the (game, walkthrough_step)'s that
        # are no longer part of the batch
        for gw in old_gw_set:
            node_ids, edge_ids = self.used_ids[gw]
            self.unused_node_ids.extend(sorted(local for _, local in node_ids.items()))
            self.unused_edge_ids.extend(sorted(local for _, local in edge_ids.items()))
            del self.used_ids[gw]

        # now update the (game, walkthrough_step)'s in the intersection
        for gw in common_gw_set:
            node_ids, edge_ids = self.used_ids[gw]
            global_nids = self.local_node_ids[gw]
            global_eids = self.local_edges[gw].keys()

            # take care of remove node ids by adding them to unused_node_ids
            # and removing them from used_ids
            for removed_node_id in sorted(node_ids.keys() - global_nids):
                self.unused_node_ids.append(node_ids[removed_node_id])
                del node_ids[removed_node_id]

            # take care of added node ids
            for added_node_id in sorted(global_nids - node_ids.keys()):
                node_ids[added_node_id] = self.unused_node_ids.popleft()

            # do the same for edges
            for removed_edge_id in sorted(edge_ids.keys() - global_eids):
                self.unused_edge_ids.append(edge_ids[removed_edge_id])
                del edge_ids[removed_edge_id]

            for added_edge_id in sorted(global_eids - edge_ids.keys()):
                edge_ids[added_edge_id] = self.unused_edge_ids.popleft()

        # now allocate the node and edge IDs of the new (game, walkthrough_step)'s
        for gw in new_gw_set:
            global_nids = self.local_node_ids[gw]
            global_eids = self.local_edges[gw].keys()
            self.used_ids[gw] = (
                dict(
                    zip(
                        sorted(global_nids),
                        (
                            self.unused_node_ids.popleft()
                            for _ in range(len(global_nids))
                        ),
                    ),
                ),
                dict(
                    zip(
                        sorted(global_eids),
                        (
                            self.unused_edge_ids.popleft()
                            for _ in range(len(global_eids))
                        ),
                    )
                ),
            )

    def collate_textual_inputs(
        self, batch: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """
        Collate the textual inputs (observation and previous action) of the given batch.

        output: {
            "obs_word_ids": (batch, obs_len),
            "obs_mask": (batch, obs_len),
            "prev_action_word_ids": (batch, prev_action_len),
            "prev_action_mask": (batch, prev_action_len),
        }
        """
        # textual observation
        obs_word_ids, obs_mask = self.preprocessor.preprocess_tokenized(
            [example["observation"].split() for example in batch]
        )

        # textual previous action
        prev_action_word_ids, prev_action_mask = self.preprocessor.preprocess_tokenized(
            [example["previous_action"].split() for example in batch]
        )
        return {
            "obs_word_ids": obs_word_ids,
            "obs_mask": obs_mask,
            "prev_action_word_ids": prev_action_word_ids,
            "prev_action_mask": prev_action_mask,
        }

    def collate_non_graphical_inputs(
        self, batch: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """
        Collate the non-graphical inputs of the given batch.

        output: {
            "tgt_event_type_ids": (event_seq_len),
            "groundtruth_event_type_ids": (event_seq_len),
            "tgt_event_timestamps": (event_seq_len),
            "tgt_event_mask": (event_seq_len),
            "tgt_event_src_mask": (event_seq_len),
            "tgt_event_dst_mask": (event_seq_len),
            "groundtruth_event_mask": (event_seq_len),
            "groundtruth_event_src_mask": (event_seq_len),
            "groundtruth_event_dst_mask": (event_seq_len),
            "tgt_event_label_ids": (event_seq_len),
            "groundtruth_event_label_ids": (event_seq_len),
        }
        """
        # event types
        event_type_ids = [
            EVENT_TYPE_ID_MAP[event["type"]]
            for example in batch
            for event in example["event_seq"]
        ]
        # prepend a start event
        tgt_event_type_ids = torch.tensor([EVENT_TYPE_ID_MAP["start"]] + event_type_ids)
        # append an end event
        groundtruth_event_type_ids = torch.tensor(
            event_type_ids + [EVENT_TYPE_ID_MAP["end"]]
        )

        tgt_event_timestamps = torch.tensor(
            # 0 timestamp for start event
            [0.0]
            + [
                float(event["timestamp"])
                for example in batch
                for event in example["event_seq"]
            ]
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

        event_label_ids = [
            self.label_id_map[event["label"]]
            for example in batch
            for event in example["event_seq"]
        ]
        # prepend a pad label for start event
        tgt_event_label_ids = torch.tensor([self.label_id_map[""]] + event_label_ids)
        # append a pad label for end event
        groundtruth_event_label_ids = torch.tensor(
            event_label_ids + [self.label_id_map[""]]
        )

        return {
            "tgt_event_type_ids": tgt_event_type_ids,
            "groundtruth_event_type_ids": groundtruth_event_type_ids,
            "tgt_event_timestamps": tgt_event_timestamps,
            "tgt_event_mask": tgt_event_mask,
            "tgt_event_src_mask": tgt_event_src_mask,
            "tgt_event_dst_mask": tgt_event_dst_mask,
            "groundtruth_event_mask": groundtruth_event_mask,
            "groundtruth_event_src_mask": groundtruth_event_src_mask,
            "groundtruth_event_dst_mask": groundtruth_event_dst_mask,
            "tgt_event_label_ids": tgt_event_label_ids,
            "groundtruth_event_label_ids": groundtruth_event_label_ids,
        }

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

    def collate_graphical_inputs(
        self, graph: TextWorldGraph, batch: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """
        Collate the graphical inputs of the given batch.

        output: {
            "node_ids": (num_nodes),
            "edge_ids": (num_edges),
            "edge_index": (2, num_edges),
            "edge_timestamps": (num_edges),
            "tgt_event_src_ids": (event_seq_len),
            "tgt_event_dst_ids": (event_seq_len),
            "tgt_event_edge_ids": (event_seq_len),
            "groundtruth_event_src_ids": (event_seq_len),
            "groundtruth_event_dst_ids": (event_seq_len),
        }
        """
        # update the graph with the graph events in the batch
        for example in batch:
            graph.process_events(
                example["event_seq"],
                game=example["game"],
                walkthrough_step=example["walkthrough_step"],
            )

        # get the subgraph node/edge ID maps based on the updated graph
        node_id_map, edge_id_map = self.calculate_subgraph_maps(graph, batch)

        # create a map from local node ID => position in node_ids
        # which will be used for edge indices and groundtruth node IDs
        node_id_pos_map = {
            node_id: i for i, node_id in enumerate(sorted(node_id_map.values()))
        }

        # collect global event node/edge IDs
        global_event_src_ids: List[int] = []
        global_event_dst_ids: List[int] = []
        global_event_edge_ids: List[int] = []
        for example in batch:
            for event in example["event_seq"]:
                if event["type"] in {"node-add", "node-delete"}:
                    global_event_src_ids.append(event["node_id"])
                    # use -1 as a placeholder. it will be translated to local 0 later
                    global_event_dst_ids.append(-1)
                    global_event_edge_ids.append(-1)
                else:
                    global_event_src_ids.append(event["src_id"])
                    global_event_dst_ids.append(event["dst_id"])
                    global_event_edge_ids.append(event["edge_id"])

        # translate global IDs to subgraph IDs
        # source IDs are never masked, so should always be available in node_id_map
        event_src_ids = [node_id_map[i] for i in global_event_src_ids]
        event_dst_ids = [0 if i == -1 else node_id_map[i] for i in global_event_dst_ids]
        event_edge_ids = [
            0 if i == -1 else edge_id_map[i] for i in global_event_edge_ids
        ]

        # placeholder node id for the start event
        tgt_event_src_ids = torch.tensor([0] + event_src_ids)
        tgt_event_dst_ids = torch.tensor([0] + event_dst_ids)

        # placeholder edge id for the start event
        tgt_event_edge_ids = torch.tensor([0] + event_edge_ids)

        # translate node IDs into the positions in node_ids
        # this is necessary to calculate cross entropy losses
        # placeholder node id 0 for the end event, which is masked in loss calculation
        groundtruth_event_src_ids = torch.tensor(
            [
                0 if nid == -1 else node_id_pos_map[node_id_map[nid]]
                for nid in global_event_src_ids
            ]
            + [0]
        )
        groundtruth_event_dst_ids = torch.tensor(
            [
                0 if nid == -1 else node_id_pos_map[node_id_map[nid]]
                for nid in global_event_dst_ids
            ]
            + [0]
        )

        # Figure out the latest timestamp for each (game, walkthrough_step) pair
        # since we want to calculate the time embeddings at the latest timestamp
        game_walkthrough_timestamp_dict: Dict[Tuple[str, int], int] = {}
        for example in batch:
            key = (example["game"], example["walkthrough_step"])
            if key not in game_walkthrough_timestamp_dict:
                game_walkthrough_timestamp_dict[key] = example["timestamp"]
            elif game_walkthrough_timestamp_dict[key] < example["timestamp"]:
                game_walkthrough_timestamp_dict[key] = example["timestamp"]

        node_id_list: List[int] = []
        edge_index_list: List[Tuple[int, int]] = []
        edge_id_list: List[int] = []
        edge_timestamps: List[int] = []
        for key, timestamp in game_walkthrough_timestamp_dict.items():
            (
                subgraph_node_ids,
                subgraph_edge_ids,
                subgraph_edge_index,
            ) = graph.get_subgraph(key)
            node_id_list.extend(node_id_map[i] for i in subgraph_node_ids)
            edge_id_list.extend(edge_id_map[i] for i in subgraph_edge_ids)
            edge_index_list.extend(
                (
                    node_id_pos_map[node_id_map[src]],
                    node_id_pos_map[node_id_map[dst]],
                )
                for src, dst in subgraph_edge_index
            )
            edge_timestamps.extend([timestamp] * len(subgraph_edge_ids))

        node_ids = torch.tensor(node_id_list)
        edge_ids = torch.tensor(edge_id_list)
        edge_index = torch.tensor(edge_index_list).transpose(0, 1)

        return {
            "node_ids": node_ids,
            "edge_ids": edge_ids,
            "edge_index": edge_index,
            "edge_timestamps": torch.tensor(edge_timestamps, dtype=torch.float),
            "tgt_event_src_ids": tgt_event_src_ids,
            "tgt_event_dst_ids": tgt_event_dst_ids,
            "tgt_event_edge_ids": tgt_event_edge_ids,
            "groundtruth_event_src_ids": groundtruth_event_src_ids,
            "groundtruth_event_dst_ids": groundtruth_event_dst_ids,
        }

    def collate(
        self, graph: TextWorldGraph, batch: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """
        Turn a batch of raw datapoints into tensors.
        {
            "obs_word_ids": (batch, obs_len),
            "obs_mask": (batch, obs_len),
            "prev_action_word_ids": (batch, prev_action_len),
            "prev_action_mask": (batch, prev_action_len),
            "node_ids": (num_nodes),
            "edge_ids": (num_edges),
            "edge_index": (2, num_edges),
            "edge_timestamps": (num_edges),
            "tgt_event_timestamps": (event_seq_len),
                Used for teacher forcing message calculation.
            "tgt_event_mask": (event_seq_len),
                Used for teacher forcing message calculation, and label mask.
            "tgt_event_type_ids": (event_seq_len),
            "tgt_event_src_ids": (event_seq_len),
            "tgt_event_src_mask": (event_seq_len),
            "tgt_event_dst_ids": (event_seq_len),
            "tgt_event_dst_mask": (event_seq_len),
            "tgt_event_edge_ids": (event_seq_len),
            "tgt_event_label_ids": (event_seq_len),
            "groundtruth_event_type_ids": (event_seq_len),
            "groundtruth_event_src_ids": (event_seq_len),
            "groundtruth_event_src_mask": (event_seq_len),
            "groundtruth_event_dst_ids": (event_seq_len),
            "groundtruth_event_dst_mask": (event_seq_len),
            "groundtruth_event_label_ids": (event_seq_len),
            "groundtruth_event_mask": (event_seq_len),
        }
        """
        results = self.collate_textual_inputs(batch)
        results.update(self.collate_non_graphical_inputs(batch))
        results.update(self.collate_graphical_inputs(graph, batch))
        return results

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.train_batch_size,
            collate_fn=partial(self.collate, TextWorldGraph()),
            pin_memory=True,
            num_workers=1,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid,
            batch_size=self.val_batch_size,
            collate_fn=partial(self.collate, TextWorldGraph()),
            pin_memory=True,
            num_workers=1,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            batch_size=self.val_batch_size,
            collate_fn=partial(self.collate, TextWorldGraph()),
            pin_memory=True,
            num_workers=1,
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
