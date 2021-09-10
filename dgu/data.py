import json
import pytorch_lightning as pl
import torch
import networkx as nx

from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from hydra.utils import to_absolute_path
from dataclasses import dataclass, field
from torch_geometric.data import Data, Batch
from copy import deepcopy

from dgu.preprocessor import SpacyPreprocessor
from dgu.nn.utils import compute_masks_from_event_type_ids
from dgu.constants import EVENT_TYPE_ID_MAP
from dgu.graph import process_triplet_cmd


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
    node_label_ids: torch.Tensor = field(default_factory=empty_tensor)
    node_memory_update_index: torch.Tensor = field(default_factory=empty_tensor)
    node_memory_update_mask: torch.Tensor = field(default_factory=empty_tensor)
    edge_index: torch.Tensor = field(default_factory=empty_tensor)
    edge_label_ids: torch.Tensor = field(default_factory=empty_tensor)
    edge_last_update: torch.Tensor = field(default_factory=empty_tensor)
    batch: torch.Tensor = field(default_factory=empty_tensor)
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
    data: Tuple[
        Tuple[
            TWCmdGenTemporalStepInput,
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
            event_src_index=event_src_index,
            event_dst_index=event_dst_index,
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
        self, obs: List[str], prev_actions: List[str], timestamps: List[int]
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

    def collate_graphical_inputs(
        self, batch_graphs: List[nx.DiGraph], batch_step: List[Dict[str, Any]]
    ) -> Tuple[List[TWCmdGenTemporalGraphicalInput], List[nx.DiGraph]]:
        """
        Collate the graphical inputs of the given batch.

        output: (
            len([TWCmdGenTemporalGraphicalInput(
                node_label_ids: (num_nodes),
                node_memory_update_index: (prev_num_nodes),
                node_memory_update_mask: (prev_num_nodes),
                edge_index: (2, num_edges),
                edge_label_ids: (num_edges),
                edge_last_update: (num_edges),
                batch: (num_nodes),
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
            ), ...]) = event_seq_len,
            updated_batch_graphs
        )
        """
        assert len(batch_graphs) == len(batch_step)
        batch_event_seq: List[List[Dict[str, Any]]] = []
        collated: List[TWCmdGenTemporalGraphicalInput] = []
        updated_batch_graphs: List[nx.DiGraph] = []
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
                updated_batch_graphs.append(deepcopy(graph))
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
                    "before_graph": curr_graph,
                    "after_graph": curr_graph,
                }
            ]
            batch_event_seq.append(event_seq)
            updated_batch_graphs.append(curr_graph)

        max_event_seq_len = max(len(event_seq) for event_seq in batch_event_seq)
        batch_size = len(batch_step)

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
            batch_event_timestamps: List[int] = []
            batch_graph_data: List[TWCmdGenTemporalGraphData] = []
            for event_seq in batch_event_seq:
                # collect event data for all the items in the batch at event i
                if seq_step_num < len(event_seq):
                    event = event_seq[seq_step_num]
                    batch_event_type_ids.append(EVENT_TYPE_ID_MAP[event["type"]])
                    event_src_id = event.get("src_id", event.get("node_id", 0))
                    batch_event_src_ids.append(event_src_id)
                    event_dst_id = event.get("dst_id", 0)
                    batch_event_dst_ids.append(event_dst_id)
                    batch_event_label_ids.append(self.label_id_map[event["label"]])
                    batch_event_timestamps.append(event["timestamp"])
                    batch_graph_data.append(
                        TWCmdGenTemporalGraphData.from_graph_event(
                            torch.tensor(event_src_id),
                            torch.tensor(event_dst_id),
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
                    # we're done for this step for this batch item,
                    # so keep the graph around for the next step
                    batch_graph_data.append(
                        TWCmdGenTemporalGraphData.from_graph_event(
                            torch.tensor(0),
                            torch.tensor(0),
                            event_seq[-1]["after_graph"],
                            event_seq[-1]["after_graph"],
                            self.label_id_map,
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
                    batch=graph_batch.batch,
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
            tgt_event_src_ids = graph_batch.event_src_index
            tgt_event_dst_ids = graph_batch.event_dst_index
            tgt_event_label_ids = groundtruth_event_label_ids
        return collated, updated_batch_graphs

    def __call__(self, batch: List[List[Dict[str, Any]]]) -> TWCmdGenTemporalBatch:
        """
        Each element in the collated batch is a batched step. Each step
        has a dictionary of textual inputs and another dictionary
        of graph event inputs.
        (
            (
                TWCmdGenTemporalStepInput(
                    obs_word_ids: (batch, obs_len),
                    obs_mask: (batch, obs_len),
                    prev_action_word_ids: (batch, prev_action_len),
                    prev_action_mask: (batch, prev_action_len),
                    timestamps: (batch)
                ),
                (
                    TWCmdGenTemporalGraphicalInput(
                        node_label_ids: (num_nodes),
                        node_memory_update_index: (prev_num_nodes),
                        node_memory_update_mask: (prev_num_nodes),
                        edge_index: (2, num_edges),
                        edge_label_ids: (num_edges),
                        edge_last_update: (num_edges),
                        batch: (num_nodes),
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
                (
                    (commands, ...),
                    ...
                )
            ),
            ...
        )
        """
        batch_graphs = [
            nx.DiGraph(
                game=steps[0]["game"], walkthrough_step=steps[0]["walkthrough_step"]
            )
            for steps in batch
        ]
        max_episode_len = max(len(episode) for episode in batch)
        collated_batch: List[
            Tuple[
                TWCmdGenTemporalStepInput,
                List[TWCmdGenTemporalGraphicalInput],
                Tuple[Tuple[str, ...], ...],
            ]
        ] = []
        updated_batch_graphs = batch_graphs
        for i in range(max_episode_len):
            batch_ith_step = [
                episode[i] if i < len(episode) else {} for episode in batch
            ]
            step = self.collate_step_inputs(
                # use "<bos> <eos>" for empty observations and previous actions
                # to prevent nan's when encoding.
                [s_i.get("observation", "<bos> <eos>") for s_i in batch_ith_step],
                [s_i.get("previous_action", "<bos> <eos>") for s_i in batch_ith_step],
                [s_i.get("timestamp", 0) for s_i in batch_ith_step],
            )
            graph_events, updated_batch_graphs = self.collate_graphical_inputs(
                updated_batch_graphs, batch_ith_step
            )
            cmds = tuple(
                tuple(step.get("target_commands", [])) for step in batch_ith_step
            )
            collated_batch.append((step, graph_events, cmds))
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
