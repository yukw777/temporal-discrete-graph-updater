import itertools
import json
import pytorch_lightning as pl
import torch

from tqdm import tqdm
from typing import List, Iterator, Dict, Any, Optional
from torch.utils.data import Sampler, Dataset, DataLoader
from hydra.utils import to_absolute_path
from functools import partial

from dgu.graph import TextWorldGraph
from dgu.preprocessor import SpacyPreprocessor
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

    This dataset also has a global graph that contains all the knowledge graphs
    for the games. This can be used to generate node and edge labels for batches.
    """

    def __init__(self, path: str) -> None:
        with open(path, "r") as f:
            raw_data = json.load(f)
        self.graph = TextWorldGraph()
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
            walkthrough_step, random_step = example["step"]
            if curr_walkthrough_step != walkthrough_step:
                # a new walkthrough step has been taken
                # add the walkthrough examples so far
                for timestamp, w_example in enumerate(walkthrough_examples):
                    event_seq = list(
                        itertools.chain.from_iterable(
                            self.graph.process_triplet_cmd(
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
                            "event_seq": event_seq,
                        }
                    )

                # add the random examples
                for timestamp, r_example in enumerate(random_examples):
                    event_seq = list(
                        itertools.chain.from_iterable(
                            self.graph.process_triplet_cmd(
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
                    self.graph.process_triplet_cmd(
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
                    "event_seq": event_seq,
                }
            )

        # add the random examples
        for timestamp, r_example in enumerate(random_examples):
            event_seq = list(
                itertools.chain.from_iterable(
                    self.graph.process_triplet_cmd(
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
                    "event_seq": event_seq,
                }
            )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]

    def __len__(self) -> int:
        return len(self.data)


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
    ) -> None:
        super().__init__()
        self.train_path = to_absolute_path(train_path)
        self.train_batch_size = train_batch_size
        self.train_num_workers = train_num_workers
        self.val_path = to_absolute_path(val_path)
        self.val_batch_size = val_batch_size
        self.val_num_workers = val_num_workers
        self.test_path = to_absolute_path(test_path)
        self.test_batch_size = test_batch_size
        self.test_num_workers = test_num_workers

        self.preprocessor = SpacyPreprocessor.load_from_file(
            to_absolute_path(word_vocab_file)
        )

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train = TWCmdGenTemporalDataset(self.train_path)
            self.valid = TWCmdGenTemporalDataset(self.val_path)

        if stage == "test" or stage is None:
            self.test = TWCmdGenTemporalDataset(self.test_path)

    def collate(
        self, stage: str, batch: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """
        Turn a batch of raw datapoints into tensors.
        {
            "obs_word_ids": (batch, obs_len),
            "obs_mask": (batch, obs_len),
            "prev_action_word_ids": (batch, prev_action_len),
            "prev_action_mask": (batch, prev_action_len),
            "subgraph_node_ids": (num_nodes),
            "event_type_ids": (event_seq_len),
            "event_timestamps": (event_seq_len),
            "event_src_ids": (event_seq_len),
            "event_src_mask": (event_seq_len),
            "event_dst_ids": (event_seq_len),
            "event_dst_mask": (event_seq_len),
            "event_label_word_ids": (event_seq_len, label_len),
            "event_label_mask": (event_seq_len, label_len),
        }
        """
        obs_word_ids, obs_mask = self.preprocessor.preprocess_tokenized(
            [example["observation"].split() for example in batch]
        )
        prev_action_word_ids, prev_action_mask = self.preprocessor.preprocess_tokenized(
            [example["previous_action"].split() for example in batch]
        )
        event_label_word_ids, event_label_mask = self.preprocessor.preprocess_tokenized(
            # empty labels for start and end tokens
            [[]]  # type: ignore
            + list(
                itertools.chain.from_iterable(
                    [
                        [event["label"].split() for event in example["event_seq"]]
                        for example in batch
                    ]
                )
            )
            + [[]]
        )
        event_type_ids = torch.tensor(
            # start and end tokens
            [EVENT_TYPE_ID_MAP["start"]]
            + [
                EVENT_TYPE_ID_MAP[event["type"]]
                for example in batch
                for event in example["event_seq"]
            ]
            + [EVENT_TYPE_ID_MAP["end"]]
        )
        event_timestamps = torch.tensor(
            # 0 timestamp for start and end tokens
            [0.0]
            + [
                float(event["timestamp"])
                for example in batch
                for event in example["event_seq"]
            ]
            + [0.0]
        )

        # mask out source IDs for the start token
        event_src_ids: List[int] = [0]
        event_src_mask: List[float] = [0.0]
        event_dst_ids: List[int] = [0]
        event_dst_mask: List[float] = [0.0]

        for example in batch:
            for event in example["event_seq"]:
                if event["type"] in {"node-add", "node-delete"}:
                    event_src_ids.append(event["node_id"])
                    if event["type"] == "node-add":
                        # if it's node-add, we mask it to zero out the previous memory
                        event_src_mask.append(0.0)
                    else:
                        event_src_mask.append(1.0)
                    event_dst_ids.append(0)
                    event_dst_mask.append(0.0)
                else:
                    event_src_ids.append(event["src_id"])
                    event_src_mask.append(1.0)
                    event_dst_ids.append(event["dst_id"])
                    event_dst_mask.append(1.0)

        # mask out the destination IDs for the end token
        event_src_ids.append(0)
        event_src_mask.append(0.0)
        event_dst_ids.append(0)
        event_dst_mask.append(0.0)

        # subgraph node IDs
        game_walkthrough_set = set(
            (example["game"], example["walkthrough_step"]) for example in batch
        )

        if stage == "train":
            graph = self.train.graph
        elif stage == "val":
            graph = self.valid.graph
        elif stage == "test":
            graph = self.test.graph
        else:
            raise ValueError(f"Unknown stage: {stage}")

        subgraph_node_ids = torch.tensor(
            list(graph.filter_node_ids(game_walkthrough_set))
        )

        return {
            "obs_word_ids": obs_word_ids,
            "obs_mask": obs_mask,
            "prev_action_word_ids": prev_action_word_ids,
            "prev_action_mask": prev_action_mask,
            "subgraph_node_ids": subgraph_node_ids,
            "event_type_ids": event_type_ids,
            "event_timestamps": event_timestamps,
            "event_src_ids": torch.tensor(event_src_ids),
            "event_src_mask": torch.tensor(event_src_mask),
            "event_dst_ids": torch.tensor(event_dst_ids),
            "event_dst_mask": torch.tensor(event_dst_mask),
            "event_label_word_ids": event_label_word_ids,
            "event_label_mask": event_label_mask,
        }

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.train_batch_size,
            collate_fn=partial(self.collate, "train"),
            pin_memory=True,
            num_workers=self.train_num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid,
            batch_size=self.val_batch_size,
            collate_fn=partial(self.collate, "val"),
            pin_memory=True,
            num_workers=self.val_num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            batch_size=self.val_batch_size,
            collate_fn=partial(self.collate, "test"),
            pin_memory=True,
            num_workers=self.val_num_workers,
        )
