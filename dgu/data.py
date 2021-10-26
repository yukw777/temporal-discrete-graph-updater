import json
import pytorch_lightning as pl
import torch
import networkx as nx

from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from hydra.utils import to_absolute_path
from dataclasses import dataclass, field
from torch_geometric.data import Batch

from dgu.preprocessor import SpacyPreprocessor
from dgu.nn.utils import compute_masks_from_event_type_ids, update_batched_graph
from dgu.constants import EVENT_TYPE_ID_MAP
from dgu.graph import process_triplet_cmd


class TWCmdGenTemporalDataset(Dataset):
    """
    TextWorld Command Generation temporal graph event dataset.

    Each data point contains the following information:
        {
            "game": "game name",
            "walkthrough_step": walkthrough step,
            "random_step": random step,
            "observation": "observation...",
            "previous_action": "previous action...",
            "timestamp": timestamp,
            "target_commands": [graph commands, ...],
            "graph_events": [graph events, ...],
            "prev_graph_events": [prev graph events, ...],
        }

    See dgu.graph for details on the graph event format. Note that graph events
    under prev_graph_events contain timestamp information too.
    """

    def __init__(self, path: str) -> None:
        with open(path, "r") as f:
            raw_data = json.load(f)
        self.walkthrough_examples: Dict[Tuple[str, int], Dict[str, Any]] = {}
        self.walkthrough_example_ids: List[Tuple[str, int]] = []
        self.random_examples: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(
            list
        )
        self.idx_map: List[Tuple[str, int, int]] = []

        for example in raw_data["examples"]:
            game = example["game"]
            walkthrough_step, random_step = example["step"]
            self.idx_map.append((game, walkthrough_step, random_step))
            if random_step == 0:
                # walkthrough example
                self.walkthrough_examples[(game, walkthrough_step)] = example
                self.walkthrough_example_ids.append((game, walkthrough_step))
            else:
                # random example
                self.random_examples[(game, walkthrough_step)].append(example)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        game, walkthrough_step, random_step = self.idx_map[idx]
        walkthrough_examples = [
            self.walkthrough_examples[(game, i)] for i in range(walkthrough_step + 1)
        ]
        random_examples = self.random_examples[(game, walkthrough_step)][:random_step]
        game_steps = walkthrough_examples + random_examples
        prev_graph_events: List[Dict[str, Any]] = []
        graph = nx.DiGraph()
        for timestamp, example in enumerate(game_steps):
            graph_events: List[Dict[str, Any]] = []
            for cmd in example["target_commands"]:
                sub_event_seq = process_triplet_cmd(graph, timestamp, cmd)
                graph_events.extend(sub_event_seq)
            if timestamp == len(game_steps) - 1:
                # last step so break
                break
            else:
                # set the timestamps and add them to prev_graph_events
                for event in graph_events:
                    event["timestamp"] = timestamp
                prev_graph_events.extend(graph_events)
        return {
            "game": game,
            "walkthrough_step": walkthrough_step,
            "random_step": random_step,
            "observation": example["observation"],
            "previous_action": example["previous_action"],
            "timestamp": timestamp,
            "target_commands": example["target_commands"],
            "graph_events": graph_events,
            "prev_graph_events": prev_graph_events,
        }

    def __len__(self) -> int:
        return len(self.idx_map)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, TWCmdGenTemporalDataset):
            return False
        return (
            self.walkthrough_examples == o.walkthrough_examples
            and self.walkthrough_example_ids == o.walkthrough_example_ids
            and self.random_examples == o.random_examples
            and self.idx_map == o.idx_map
        )


def empty_tensor() -> torch.Tensor:
    return torch.empty(0)


@dataclass(frozen=True)
class TWCmdGenTemporalStepInput:
    obs_word_ids: torch.Tensor = field(default_factory=empty_tensor)
    obs_mask: torch.Tensor = field(default_factory=empty_tensor)
    prev_action_word_ids: torch.Tensor = field(default_factory=empty_tensor)
    prev_action_mask: torch.Tensor = field(default_factory=empty_tensor)
    timestamps: torch.Tensor = field(default_factory=empty_tensor)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, TWCmdGenTemporalStepInput):
            return False
        return all(getattr(self, f).equal(getattr(o, f)) for f in self.__annotations__)

    def to(self, *args, **kwargs) -> "TWCmdGenTemporalStepInput":
        return TWCmdGenTemporalStepInput(
            **{f: getattr(self, f).to(*args, **kwargs) for f in self.__annotations__}
        )

    def pin_memory(self) -> "TWCmdGenTemporalStepInput":
        return TWCmdGenTemporalStepInput(
            **{f: getattr(self, f).pin_memory() for f in self.__annotations__}
        )


@dataclass(frozen=True)
class TWCmdGenTemporalGraphicalInput:
    tgt_event_type_ids: torch.Tensor = field(default_factory=empty_tensor)
    tgt_event_src_ids: torch.Tensor = field(default_factory=empty_tensor)
    tgt_event_dst_ids: torch.Tensor = field(default_factory=empty_tensor)
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
    ids: Tuple[Tuple[str, int, int], ...]
    step_input: TWCmdGenTemporalStepInput
    graphical_input_seq: Tuple[TWCmdGenTemporalGraphicalInput, ...]
    prev_batched_graph: Batch
    graph_commands: Tuple[Tuple[str, ...], ...]

    def to(self, *args, **kwargs) -> "TWCmdGenTemporalBatch":
        return TWCmdGenTemporalBatch(
            ids=self.ids,
            step_input=self.step_input.to(*args, **kwargs),
            graphical_input_seq=tuple(
                graphical.to(*args, **kwargs) for graphical in self.graphical_input_seq
            ),
            prev_batched_graph=self.prev_batched_graph.to(*args, **kwargs),
            graph_commands=self.graph_commands,
        )

    def pin_memory(self) -> "TWCmdGenTemporalBatch":
        return TWCmdGenTemporalBatch(
            ids=self.ids,
            step_input=self.step_input.pin_memory(),
            graphical_input_seq=tuple(
                graphical.pin_memory() for graphical in self.graphical_input_seq
            ),
            prev_batched_graph=self.prev_batched_graph.pin_memory(),
            graph_commands=self.graph_commands,
        )


class TWCmdGenTemporalDataCollator:
    def __init__(
        self,
        preprocessor: SpacyPreprocessor,
        label_id_map: Dict[str, int],
    ) -> None:
        self.preprocessor = preprocessor
        self.label_id_map = label_id_map

    def collate_step_inputs(
        self,
        obs: List[str],
        prev_actions: List[str],
        timestamps: List[int],
    ) -> TWCmdGenTemporalStepInput:
        """
        Collate step data such as observation, previous action and timestamp.

        output: TWCmdGenTemporalStepInput(
            obs_word_ids: (batch, obs_len),
            obs_mask: (batch, obs_len),
            prev_action_word_ids: (batch, prev_action_len),
            prev_action_mask: (batch, prev_action_len),
            timestamps: (batch)
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
        return TWCmdGenTemporalStepInput(
            obs_word_ids=obs_word_ids,
            obs_mask=obs_mask,
            prev_action_word_ids=prev_action_word_ids,
            prev_action_mask=prev_action_mask,
            timestamps=torch.tensor(timestamps, dtype=torch.float),
        )

    def collate_graphical_input_seq(
        self, batch: List[Dict[str, Any]]
    ) -> Tuple[TWCmdGenTemporalGraphicalInput, ...]:
        """
        Collate the graphical input sequence of the given batch.

        output: len([TWCmdGenTemporalGraphicalInput(
            tgt_event_type_ids: (batch),
            tgt_event_src_ids: (batch),
            tgt_event_dst_ids: (batch),
            tgt_event_label_ids: (batch),
            groundtruth_event_type_ids: (batch),
            groundtruth_event_src_ids: (batch),
            groundtruth_event_src_mask: (batch), boolean
            groundtruth_event_dst_ids: (batch),
            groundtruth_event_dst_mask: (batch), boolean
            groundtruth_event_label_ids: (batch),
            groundtruth_event_mask: (batch), boolean
        ), ...]) = event_seq_len
        """
        batch_event_seq: List[List[Dict[str, Any]]] = []
        collated: List[TWCmdGenTemporalGraphicalInput] = []
        for step in batch:
            batch_event_seq.append(
                step["graph_events"] + [{"type": "end", "label": ""}]
            )

        max_event_seq_len = max(len(event_seq) for event_seq in batch_event_seq)
        batch_size = len(batch)

        # left shifted target events
        tgt_event_type_ids = torch.tensor([EVENT_TYPE_ID_MAP["start"]] * batch_size)
        tgt_event_src_ids = torch.tensor([0] * batch_size)
        tgt_event_dst_ids = torch.tensor([0] * batch_size)
        tgt_event_label_ids = torch.tensor([self.label_id_map[""]] * batch_size)
        for seq_step_num in range(max_event_seq_len):
            batch_event_type_ids: List[int] = []
            batch_event_src_ids: List[int] = []
            batch_event_dst_ids: List[int] = []
            batch_event_label_ids: List[int] = []
            for event_seq in batch_event_seq:
                # collect event data for all the items in the batch at event i
                if seq_step_num < len(event_seq):
                    event = event_seq[seq_step_num]
                    batch_event_type_ids.append(EVENT_TYPE_ID_MAP[event["type"]])
                    batch_event_src_ids.append(
                        event.get("src_id", event.get("node_id", 0))
                    )
                    batch_event_dst_ids.append(event.get("dst_id", 0))
                    batch_event_label_ids.append(self.label_id_map[event["label"]])
                else:
                    batch_event_type_ids.append(EVENT_TYPE_ID_MAP["pad"])
                    batch_event_src_ids.append(0)
                    batch_event_dst_ids.append(0)
                    batch_event_label_ids.append(self.label_id_map[""])
            groundtruth_event_type_ids = torch.tensor(batch_event_type_ids)
            groundtruth_event_src_ids = torch.tensor(batch_event_src_ids)
            groundtruth_event_dst_ids = torch.tensor(batch_event_dst_ids)
            groundtruth_event_label_ids = torch.tensor(batch_event_label_ids)
            (
                groundtruth_event_mask,
                groundtruth_event_src_mask,
                groundtruth_event_dst_mask,
            ) = compute_masks_from_event_type_ids(groundtruth_event_type_ids)
            collated.append(
                TWCmdGenTemporalGraphicalInput(
                    tgt_event_type_ids=tgt_event_type_ids,
                    tgt_event_src_ids=tgt_event_src_ids,
                    tgt_event_dst_ids=tgt_event_dst_ids,
                    tgt_event_label_ids=tgt_event_label_ids,
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

        return tuple(collated)

    def collate_prev_graph_events(self, batch: List[Dict[str, Any]]) -> Batch:
        """
        Collate the previous graph events of the given batch into a single
        batched global graph.

        output: Batch(
            batch: (num_node)
            x: (num_node)
            node_last_update: (num_node)
            edge_index: (2, num_edge)
            edge_attr: (num_edge)
            edge_last_update: (num_edge)
        )
        """
        batch_prev_event_seq: List[List[Dict[str, Any]]] = []
        # initialize empty batch graph and memory
        batched_graph = Batch(
            batch=torch.empty(0, dtype=torch.long),
            x=torch.empty(0, dtype=torch.long),
            node_last_update=torch.empty(0),
            edge_index=torch.empty(2, 0, dtype=torch.long),
            edge_attr=torch.empty(0, dtype=torch.long),
            edge_last_update=torch.empty(0),
        )
        for step in batch:
            batch_prev_event_seq.append(step["prev_graph_events"])

        max_prev_event_seq_len = max(
            len(event_seq) for event_seq in batch_prev_event_seq
        )

        for seq_step_num in range(max_prev_event_seq_len):
            batch_event_type_ids: List[int] = []
            batch_event_src_ids: List[int] = []
            batch_event_dst_ids: List[int] = []
            batch_event_label_ids: List[int] = []
            batch_event_timestamps: List[float] = []
            for event_seq in batch_prev_event_seq:
                # collect event data for all the items in the batch at event i
                if seq_step_num < len(event_seq):
                    event = event_seq[seq_step_num]
                    batch_event_type_ids.append(EVENT_TYPE_ID_MAP[event["type"]])
                    batch_event_src_ids.append(
                        event.get("src_id", event.get("node_id", 0))
                    )
                    batch_event_dst_ids.append(event.get("dst_id", 0))
                    batch_event_label_ids.append(self.label_id_map[event["label"]])
                    batch_event_timestamps.append(float(event["timestamp"]))
                else:
                    batch_event_type_ids.append(EVENT_TYPE_ID_MAP["pad"])
                    batch_event_src_ids.append(0)
                    batch_event_dst_ids.append(0)
                    batch_event_label_ids.append(self.label_id_map[""])
                    batch_event_timestamps.append(0.0)
            batched_graph = update_batched_graph(
                batched_graph,
                torch.tensor(batch_event_type_ids),
                torch.tensor(batch_event_src_ids),
                torch.tensor(batch_event_dst_ids),
                torch.tensor(batch_event_label_ids),
                torch.tensor(batch_event_timestamps),
            )

        return batched_graph

    def __call__(self, batch: List[Dict[str, Any]]) -> TWCmdGenTemporalBatch:
        """
        Each batch has a batch of IDs, batched textual input, batched graphical
        input, batched previous graph events and ground-truth graph commands.

        TWCmdGenTemporalBatch(
            ids=(
                (game, walkthrough_step, random_step),
                ...
            ),
            step_input=TWCmdGenTemporalStepInput(
                obs_word_ids: (batch, obs_len),
                obs_mask: (batch, obs_len),
                prev_action_word_ids: (batch, prev_action_len),
                prev_action_mask: (batch, prev_action_len),
                timestamps: (batch)
            ),
            graphical_input_seq=(
                TWCmdGenTemporalGraphicalInput(
                    tgt_event_type_ids: (batch),
                    tgt_event_src_ids: (batch),
                    tgt_event_dst_ids: (batch),
                    tgt_event_label_ids: (batch),
                    groundtruth_event_type_ids: (batch),
                    groundtruth_event_src_ids: (batch),
                    groundtruth_event_src_mask: (batch),
                    groundtruth_event_dst_ids: (batch),
                    groundtruth_event_dst_mask: (batch),
                    groundtruth_event_label_ids: (batch),
                    groundtruth_event_mask: (batch),
                ),
                ...
            ),
            prev_graph_events=(
                TWCmdGenTemporalGraphEvent(
                    event_type_ids: (batch),
                    event_src_ids: (batch),
                    event_dst_ids: (batch),
                    event_label_ids: (batch),
                    event_timestamps: (batch),
                ),
                ...
            ),
            graph_commands=(
                (
                    (commands, ...),
                    ...
                ),
                ...
            )
        )
        """
        return TWCmdGenTemporalBatch(
            ids=tuple(
                (
                    str(step["game"]),
                    int(step["walkthrough_step"]),
                    int(step["random_step"]),
                )
                for step in batch
            ),
            step_input=self.collate_step_inputs(
                [step["observation"] for step in batch],
                [step["previous_action"] for step in batch],
                [step["timestamp"] for step in batch],
            ),
            graphical_input_seq=self.collate_graphical_input_seq(batch),
            prev_batched_graph=self.collate_prev_graph_events(batch),
            graph_commands=tuple(tuple(step["target_commands"]) for step in batch),
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
        word_vocab_path: str,
        node_vocab_path: str,
        relation_vocab_path: str,
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
            to_absolute_path(word_vocab_path)
        )
        self.labels, self.label_id_map = read_label_vocab_files(
            to_absolute_path(node_vocab_path), to_absolute_path(relation_vocab_path)
        )

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
                self.preprocessor, self.label_id_map
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
                self.preprocessor, self.label_id_map
            ),
            pin_memory=True,
            num_workers=self.val_num_worker,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            batch_size=self.val_batch_size,
            collate_fn=TWCmdGenTemporalDataCollator(
                self.preprocessor, self.label_id_map
            ),
            pin_memory=True,
            num_workers=self.test_num_worker,
        )


def read_label_vocab_files(
    node_vocab_path: str, relation_vocab_path: str
) -> Tuple[List[str], Dict[str, int]]:
    labels = [""]
    with open(node_vocab_path) as f:
        for line in f:
            stripped = line.strip()
            if stripped != "":
                labels.append(stripped)
    with open(relation_vocab_path) as f:
        for line in f:
            stripped = line.strip()
            if stripped != "":
                labels.append(stripped)
    return labels, {label: i for i, label in enumerate(labels)}
