import itertools
import json
import pytorch_lightning as pl
import torch
import pickle

from tqdm import tqdm
from typing import List, Iterator, Dict, Any, Optional, Tuple, Set
from torch.utils.data import Sampler, Dataset, DataLoader
from hydra.utils import to_absolute_path
from functools import partial
from pathlib import Path

from dgu.graph import TextWorldGraph
from dgu.preprocessor import SpacyPreprocessor
from dgu.nn.utils import compute_masks_from_event_type_ids, find_indices
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
        {
            "game": "game name",
            "walkthrough_step": walkthrough step,
            "observation": "observation...",
            "previous_action": "previous action...",
            "event_seq": [graph event, ...],
        }


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
        self.data: List[Dict[str, Any]] = []
        walkthrough_examples: List[Dict[str, Any]] = []
        random_examples: List[Dict[str, Any]] = []
        curr_walkthrough_step = -1
        curr_game = ""

        for example in tqdm(raw_data["examples"], desc="processing examples"):
            game = example["game"]
            if curr_game == "":
                # if it's the first game, set it right away
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
                    self.data.append(
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
                    self.data.append(
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
                    # new game, so reset walkthrough_examples
                    walkthrough_examples = []
                    curr_game = game
                walkthrough_examples.append(example)
                random_examples = []
                curr_walkthrough_step = walkthrough_step
            else:
                # a new random step has been taken, so add to random_examples
                random_examples.append(example)

        # take care of the stragglers
        # add the walkthrough examples
        for timestamp, w_example in enumerate(walkthrough_examples):
            event_seq = list(
                itertools.chain.from_iterable(
                    graph.process_triplet_cmd(
                        curr_game, curr_walkthrough_step, timestamp, cmd
                    )
                    for cmd in w_example["target_commands"]
                )
            )
            self.data.append(
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
            self.data.append(
                {
                    "game": curr_game,
                    "walkthrough_step": curr_walkthrough_step,
                    "observation": r_example["observation"],
                    "previous_action": r_example["previous_action"],
                    "timestamp": len(walkthrough_examples) + timestamp,
                    "event_seq": event_seq,
                }
            )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]

    def __len__(self) -> int:
        return len(self.data)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, TWCmdGenTemporalDataset):
            return False
        return self.data == o.data


class TWCmdGenTemporalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_path: str,
        train_batch_size: int,
        train_num_workers: int,
        val_path: str,
        val_batch_size: int,
        val_num_workers: int,
        test_path: str,
        test_batch_size: int,
        test_num_workers: int,
        word_vocab_file: str,
        node_vocab_file: str,
        relation_vocab_file: str,
    ) -> None:
        super().__init__()
        self.train_path = to_absolute_path(train_path)
        self.serialized_train_path = self.get_serialized_path(self.train_path)
        self.train_batch_size = train_batch_size
        self.train_num_workers = train_num_workers
        self.val_path = to_absolute_path(val_path)
        self.serialized_val_path = self.get_serialized_path(self.val_path)
        self.val_batch_size = val_batch_size
        self.val_num_workers = val_num_workers
        self.test_path = to_absolute_path(test_path)
        self.serialized_test_path = self.get_serialized_path(self.test_path)
        self.test_batch_size = test_batch_size
        self.test_num_workers = test_num_workers

        self.preprocessor = SpacyPreprocessor.load_from_file(
            to_absolute_path(word_vocab_file)
        )
        self.label_id_map = self.read_label_vocab_files(
            to_absolute_path(node_vocab_file), to_absolute_path(relation_vocab_file)
        )

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
            "subgraph_node_ids": (num_nodes),
            "subgraph_edge_ids": (num_edges),
            "subgraph_edge_index": (2, num_edges),
            "subgraph_edge_timestamps": (num_edges),
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
            "groundtruth_event_subgraph_src_ids": (event_seq_len),
            "groundtruth_event_src_mask": (event_seq_len),
            "groundtruth_event_dst_ids": (event_seq_len),
            "groundtruth_event_subgraph_dst_ids": (event_seq_len),
            "groundtruth_event_dst_mask": (event_seq_len),
            "groundtruth_event_label_ids": (event_seq_len),
            "groundtruth_event_mask": (event_seq_len),
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

        event_src_ids: List[int] = []
        event_dst_ids: List[int] = []
        event_edge_ids: List[int] = []

        for example in batch:
            graph.process_events(
                example["event_seq"],
                game=example["game"],
                walkthrough_step=example["walkthrough_step"],
            )
            for event in example["event_seq"]:
                if event["type"] in {"node-add", "node-delete"}:
                    event_src_ids.append(event["node_id"])
                    event_dst_ids.append(0)
                    event_edge_ids.append(0)
                else:
                    event_src_ids.append(event["src_id"])
                    event_dst_ids.append(event["dst_id"])
                    event_edge_ids.append(event["edge_id"])

        # placeholder node id for the start event
        tgt_event_src_ids = torch.tensor([0] + event_src_ids)
        tgt_event_dst_ids = torch.tensor([0] + event_dst_ids)

        # placeholder edge id for the start event
        tgt_event_edge_ids = torch.tensor([0] + event_edge_ids)

        # placeholder node id for the end event
        groundtruth_event_src_ids = torch.tensor(event_src_ids + [0])
        groundtruth_event_dst_ids = torch.tensor(event_dst_ids + [0])

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

        # calculate subgraph information so that we only calculate node embeddings
        # that are relevant.
        # First, figure out the latest timestamp for each (game, walkthrough_step) pair
        # since we want to calculate the time embeddings at the latest timestamp
        game_walkthrough_timestamp_dict: Dict[Tuple[str, int], int] = {}
        for example in batch:
            key = (example["game"], example["walkthrough_step"])
            if key not in game_walkthrough_timestamp_dict:
                game_walkthrough_timestamp_dict[key] = example["timestamp"]
            elif game_walkthrough_timestamp_dict[key] < example["timestamp"]:
                game_walkthrough_timestamp_dict[key] = example["timestamp"]

        # always include the 0th node as a placeholder
        subgraph_node_id_set: Set[int] = {0}
        subgraph_edge_ids: List[int] = []
        subgraph_edge_index_set: Set[Tuple[int, int]] = set()
        subgraph_edge_timestamps: List[int] = []
        for key, timestamp in game_walkthrough_timestamp_dict.items():
            node_ids, edge_ids, edge_index = graph.get_subgraph({key})
            subgraph_node_id_set.update(node_ids)
            subgraph_edge_ids.extend(edge_ids)
            subgraph_edge_index_set.update(edge_index)
            subgraph_edge_timestamps.extend([timestamp] * len(edge_ids))
        subgraph_node_ids = torch.tensor(sorted(subgraph_node_id_set))
        subgraph_global_edge_index = torch.tensor(
            sorted(subgraph_edge_index_set)
        ).transpose(0, 1)
        subgraph_local_edge_index = torch.stack(
            [
                find_indices(subgraph_node_ids, subgraph_global_edge_index[0]),
                find_indices(subgraph_node_ids, subgraph_global_edge_index[1]),
            ]
        )

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
            "obs_word_ids": obs_word_ids,
            "obs_mask": obs_mask,
            "prev_action_word_ids": prev_action_word_ids,
            "prev_action_mask": prev_action_mask,
            "subgraph_node_ids": subgraph_node_ids,
            "subgraph_edge_ids": torch.tensor(subgraph_edge_ids),
            "subgraph_global_edge_index": subgraph_global_edge_index,
            "subgraph_local_edge_index": subgraph_local_edge_index,
            "subgraph_edge_timestamps": torch.tensor(subgraph_edge_timestamps),
            "tgt_event_timestamps": tgt_event_timestamps,
            "tgt_event_mask": tgt_event_mask,
            "tgt_event_type_ids": tgt_event_type_ids,
            "tgt_event_src_ids": tgt_event_src_ids,
            "tgt_event_src_mask": tgt_event_src_mask,
            "tgt_event_dst_ids": tgt_event_dst_ids,
            "tgt_event_dst_mask": tgt_event_dst_mask,
            "tgt_event_edge_ids": tgt_event_edge_ids,
            "tgt_event_label_ids": tgt_event_label_ids,
            "groundtruth_event_type_ids": groundtruth_event_type_ids,
            "groundtruth_event_src_ids": groundtruth_event_src_ids,
            "groundtruth_event_subgraph_src_ids": find_indices(
                subgraph_node_ids, groundtruth_event_src_ids
            ),
            "groundtruth_event_src_mask": groundtruth_event_src_mask,
            "groundtruth_event_dst_ids": groundtruth_event_dst_ids,
            "groundtruth_event_subgraph_dst_ids": find_indices(
                subgraph_node_ids, groundtruth_event_dst_ids
            ),
            "groundtruth_event_dst_mask": groundtruth_event_dst_mask,
            "groundtruth_event_label_ids": groundtruth_event_label_ids,
            "groundtruth_event_mask": groundtruth_event_mask,
        }

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.train_batch_size,
            collate_fn=partial(self.collate, TextWorldGraph()),
            pin_memory=True,
            num_workers=self.train_num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid,
            batch_size=self.val_batch_size,
            collate_fn=partial(self.collate, TextWorldGraph()),
            pin_memory=True,
            num_workers=self.val_num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            batch_size=self.val_batch_size,
            collate_fn=partial(self.collate, TextWorldGraph()),
            pin_memory=True,
            num_workers=self.val_num_workers,
        )

    @staticmethod
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
