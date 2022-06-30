import json
import pytorch_lightning as pl
import torch
import networkx as nx
import random

from typing import List, Dict, Any, Optional, Tuple, Iterator
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, IterableDataset
from hydra.utils import to_absolute_path
from dataclasses import dataclass, field
from torch_geometric.data import Batch

from tdgu.preprocessor import Preprocessor
from tdgu.nn.utils import (
    compute_masks_from_event_type_ids,
    shift_tokens_right,
    update_batched_graph,
)
from tdgu.utils import draw_graph
from tdgu.constants import (
    EVENT_TYPE_ID_MAP,
    TWO_ARGS_RELATIONS,
    NORTH_OF,
    SOUTH_OF,
    EAST_OF,
    WEST_OF,
    PART_OF,
    IS,
    NEEDS,
)
from tdgu.graph import process_triplet_cmd


class TWCmdGenGraphEventDataset(Dataset):
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

    def __init__(
        self,
        path: str,
        allow_objs_with_same_label: bool = False,
        sort_commands: bool = False,
    ) -> None:
        self.allow_objs_with_same_label = allow_objs_with_same_label
        with open(path, "r") as f:
            raw_data = json.load(f)
        self.walkthrough_examples: Dict[Tuple[str, int], Dict[str, Any]] = {}
        self.walkthrough_example_ids: List[Tuple[str, int]] = []
        self.random_examples: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(
            list
        )
        self.idx_map: List[Tuple[str, int, int]] = []

        for example in raw_data["examples"]:
            if sort_commands:
                example["target_commands"] = sort_target_commands(
                    example["target_commands"]
                )
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

    def __getitem__(
        self, idx: int, draw_graph_filename: Optional[str] = None
    ) -> Dict[str, Any]:
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
                sub_event_seq = process_triplet_cmd(
                    graph,
                    timestamp,
                    cmd,
                    allow_objs_with_same_label=self.allow_objs_with_same_label,
                )
                graph_events.extend(sub_event_seq)
            if timestamp == len(game_steps) - 1:
                # last step so break
                break
            else:
                # set the timestamps and add them to prev_graph_events
                for i, event in enumerate(graph_events):
                    event["timestamp"] = [timestamp, i]
                prev_graph_events.extend(graph_events)
        if draw_graph_filename is not None:
            for n, data in graph.nodes.data():
                data["label"] = n.label
            draw_graph(graph, draw_graph_filename)
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
        if not isinstance(o, TWCmdGenGraphEventDataset):
            return False
        return (
            self.walkthrough_examples == o.walkthrough_examples
            and self.walkthrough_example_ids == o.walkthrough_example_ids
            and self.random_examples == o.random_examples
            and self.idx_map == o.idx_map
        )


class TWCmdGenGraphEventFreeRunDataset(IterableDataset):
    """
    TextWorld Command Generation graph event dataset for free run evaluation.

    Free run means we run the model from the beginning of the game (empty graph)
    till the end.

    This dataset is an iterable dataset that keeps a batch of walkthroughs then
    produces batches of game steps, where each game step is a dictionary:
    {
        "game": "game name",
        "walkthrough_step": walkthrough step,
        "random_step": random step,
        "observation": "observation...",
        "previous_action": "previous action...",
        "timestamp": timestamp,
        "target_commands": [graph commands, ...],
        "graph_events": [graph events, ...],
        "previous_graph_seen": [(src, dst, rel), ...],
    }
    """

    def __init__(
        self,
        path: str,
        batch_size: int,
        allow_objs_with_same_label: bool = False,
        sort_commands: bool = False,
        shuffle: bool = False,
    ) -> None:
        self.allow_objs_with_same_label = allow_objs_with_same_label
        self.shuffle = shuffle
        self.batch_size = batch_size
        with open(path, "r") as f:
            raw_data = json.load(f)
        self.walkthrough_examples: Dict[Tuple[str, int], Dict[str, Any]] = {}
        self.walkthrough_example_ids: List[Tuple[str, int]] = []
        self.random_examples: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(
            list
        )
        self.graph_index = json.loads(raw_data["graph_index"])

        for example in raw_data["examples"]:
            if sort_commands:
                example["target_commands"] = sort_target_commands(
                    example["target_commands"]
                )
            game = example["game"]
            walkthrough_step, random_step = example["step"]
            if random_step == 0:
                # walkthrough example
                self.walkthrough_examples[(game, walkthrough_step)] = example
                self.walkthrough_example_ids.append((game, walkthrough_step))
            else:
                # random example
                self.random_examples[(game, walkthrough_step)].append(example)

    def __iter__(self) -> Iterator[List[Tuple[int, Dict[str, Any]]]]:
        if self.shuffle:
            random.shuffle(self.walkthrough_example_ids)
        walkthrough_id_to_step: Dict[int, Tuple[int, List[Dict[str, Any]]]] = {}
        last_added_walkthrough_id = 0
        while (
            last_added_walkthrough_id < len(self.walkthrough_example_ids)
            or len(walkthrough_id_to_step) > 0
        ):
            # fetch a batch of walkthroughs
            while (
                last_added_walkthrough_id < len(self.walkthrough_example_ids)
                and len(walkthrough_id_to_step) < self.batch_size
            ):
                walkthrough_id_to_step[last_added_walkthrough_id] = (
                    0,
                    self._get_walkthrough(last_added_walkthrough_id),
                )
                last_added_walkthrough_id += 1
            while True:
                yield [
                    (walkthrough_id, walkthrough[step_id])
                    for walkthrough_id, (
                        step_id,
                        walkthrough,
                    ) in walkthrough_id_to_step.items()
                ]

                # update the step ids for walkthroughs
                for walkthrough_id, (
                    step_id,
                    walkthrough,
                ) in walkthrough_id_to_step.items():
                    walkthrough_id_to_step[walkthrough_id] = (step_id + 1, walkthrough)

                # remove finished walkthroughs
                finished_walkthrough_ids = [
                    walkthrough_id
                    for walkthrough_id, (step_id, _) in walkthrough_id_to_step.items()
                    if step_id >= self._get_walkthrough_len(walkthrough_id)
                ]
                for walkthrough_id in finished_walkthrough_ids:
                    walkthrough_id_to_step.pop(walkthrough_id)

                # if there are finished walkthroughs, break and fetch more.
                if len(finished_walkthrough_ids) > 0:
                    break

    def _get_walkthrough_len(self, walkthrough_id: int) -> int:
        game, walkthrough_step = self.walkthrough_example_ids[walkthrough_id]
        return (
            walkthrough_step + 1 + len(self.random_examples[(game, walkthrough_step)])
        )

    def _get_walkthrough(self, walkthrough_id: int) -> List[Dict[str, Any]]:
        game, walkthrough_step = self.walkthrough_example_ids[walkthrough_id]
        walkthrough_examples = [
            self.walkthrough_examples[(game, i)] for i in range(walkthrough_step + 1)
        ]
        random_examples = self.random_examples[(game, walkthrough_step)]
        data: List[Dict[str, Any]] = []
        graph = nx.DiGraph()
        for timestamp, example in enumerate(walkthrough_examples + random_examples):
            graph_events: List[Dict[str, Any]] = []
            for cmd in example["target_commands"]:
                sub_event_seq = process_triplet_cmd(
                    graph,
                    timestamp,
                    cmd,
                    allow_objs_with_same_label=self.allow_objs_with_same_label,
                )
                graph_events.extend(sub_event_seq)
            data.append(
                {
                    "game": game,
                    "walkthrough_step": walkthrough_step,
                    "observation": example["observation"],
                    "previous_action": example["previous_action"],
                    "timestamp": timestamp,
                    "target_commands": example["target_commands"],
                    "graph_events": graph_events,
                    "previous_graph_seen": [
                        f"{self.graph_index['entities'][str(src)]} , "
                        f"{self.graph_index['entities'][str(dst)]} , "
                        f"{self.graph_index['relation_types'][str(rel)]}"
                        for src, dst, rel in [
                            self.graph_index["relations"][str(relation_id)]
                            for relation_id in self.graph_index["graphs"][
                                str(example["previous_graph_seen"])
                            ]
                        ]
                    ],
                }
            )
        return data

    def __len__(self) -> int:
        return len(self.walkthrough_example_ids)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, TWCmdGenGraphEventFreeRunDataset):
            return False
        return (
            self.walkthrough_examples == o.walkthrough_examples
            and self.walkthrough_example_ids == o.walkthrough_example_ids
            and self.random_examples == o.random_examples
            and self.graph_index == o.graph_index
        )


class TWCmdGenObsGenDataset(Dataset):
    """
    TextWorld Command Generation observation generation dataset.
    {
        "game": "game name",
        "walkthrough_step": walkthrough step,
        "steps": [{
            "observation": "observation...",
            "previous_action": "previous action...",
        }, ...]
    }

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

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        game, walkthrough_step = self.walkthrough_example_ids[idx]
        walkthrough_examples = [
            self.walkthrough_examples[(game, i)] for i in range(walkthrough_step + 1)
        ]
        random_examples = self.random_examples[(game, walkthrough_step)]
        steps = []
        for example in walkthrough_examples + random_examples:
            steps.append(
                {
                    "observation": example["observation"],
                    "previous_action": example["previous_action"],
                }
            )
        return {"game": game, "walkthrough_step": walkthrough_step, "steps": steps}

    def __len__(self) -> int:
        return len(self.walkthrough_example_ids)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, TWCmdGenObsGenDataset):
            return False
        return (
            self.walkthrough_examples == o.walkthrough_examples
            and self.walkthrough_example_ids == o.walkthrough_example_ids
            and self.random_examples == o.random_examples
        )


def sort_target_commands(list_of_cmds: List[str]) -> List[str]:
    """
    Copied from the original GATA code
    """
    list_of_cmd_tokens = [item.split(" , ") for item in list_of_cmds]

    def key_fn(
        cmd: List[str],
    ) -> Tuple[bool, bool, bool, bool, bool, bool, bool, bool, str, str]:
        return (
            cmd[0] == "add",  # add always before delete
            cmd[1] == "player",  # relations with player always first
            cmd[2] == "player",  # relations with player always first
            cmd[3]
            in {
                WEST_OF,
                EAST_OF,
                NORTH_OF,
                SOUTH_OF,
            },  # room connections always first
            cmd[3] in {PART_OF},  # recipe
            cmd[3] in set(TWO_ARGS_RELATIONS),  # two args relations first
            cmd[3] in {IS},  # one arg state relations first
            cmd[3] in {NEEDS},  # one arg requirement relations first
            cmd[2],
            cmd[1],
        )

    list_of_cmds = [
        " , ".join(item)
        for item in sorted(list_of_cmd_tokens, key=key_fn, reverse=True)
    ]
    res: List[str] = []
    for cmd in list_of_cmds:
        if cmd not in res:
            res.append(cmd)
    return res


def empty_tensor() -> torch.Tensor:
    return torch.empty(0)


def empty_graph() -> Batch:
    return Batch(
        batch=torch.empty(0, dtype=torch.long),
        x=torch.empty(0, 0, dtype=torch.long),
        node_label_mask=torch.empty(0, 0).bool(),
        node_last_update=torch.empty(0, 2, dtype=torch.long),
        edge_index=torch.empty(2, 0, dtype=torch.long),
        edge_attr=torch.empty(0, 0).long(),
        edge_label_mask=torch.empty(0, 0).bool(),
        edge_last_update=torch.empty(0, 2, dtype=torch.long),
    )


@dataclass(frozen=True)
class TWCmdGenGraphEventStepInput:
    obs_word_ids: torch.Tensor = field(default_factory=empty_tensor)
    obs_mask: torch.Tensor = field(default_factory=empty_tensor)
    prev_action_word_ids: torch.Tensor = field(default_factory=empty_tensor)
    prev_action_mask: torch.Tensor = field(default_factory=empty_tensor)
    timestamps: torch.Tensor = field(default_factory=empty_tensor)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, TWCmdGenGraphEventStepInput):
            return False
        return all(getattr(self, f).equal(getattr(o, f)) for f in self.__annotations__)

    def to(self, *args, **kwargs) -> "TWCmdGenGraphEventStepInput":
        return TWCmdGenGraphEventStepInput(
            **{f: getattr(self, f).to(*args, **kwargs) for f in self.__annotations__}
        )

    def pin_memory(self) -> "TWCmdGenGraphEventStepInput":
        return TWCmdGenGraphEventStepInput(
            **{f: getattr(self, f).pin_memory() for f in self.__annotations__}
        )


@dataclass(frozen=True)
class TWCmdGenGraphEventGraphicalInput:
    tgt_event_type_ids: torch.Tensor = field(default_factory=empty_tensor)
    tgt_event_src_ids: torch.Tensor = field(default_factory=empty_tensor)
    tgt_event_dst_ids: torch.Tensor = field(default_factory=empty_tensor)
    tgt_event_label_word_ids: torch.Tensor = field(default_factory=empty_tensor)
    tgt_event_label_mask: torch.Tensor = field(default_factory=empty_tensor)
    groundtruth_event_type_ids: torch.Tensor = field(default_factory=empty_tensor)
    groundtruth_event_mask: torch.Tensor = field(default_factory=empty_tensor)
    groundtruth_event_src_ids: torch.Tensor = field(default_factory=empty_tensor)
    groundtruth_event_src_mask: torch.Tensor = field(default_factory=empty_tensor)
    groundtruth_event_dst_ids: torch.Tensor = field(default_factory=empty_tensor)
    groundtruth_event_dst_mask: torch.Tensor = field(default_factory=empty_tensor)
    groundtruth_event_label_tgt_word_ids: torch.Tensor = field(
        default_factory=empty_tensor
    )
    groundtruth_event_label_tgt_mask: torch.Tensor = field(default_factory=empty_tensor)
    groundtruth_event_label_groundtruth_word_ids: torch.Tensor = field(
        default_factory=empty_tensor
    )
    groundtruth_event_label_mask: torch.Tensor = field(default_factory=empty_tensor)
    prev_batched_graph: Batch = field(default_factory=empty_graph)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, TWCmdGenGraphEventGraphicalInput):
            return False
        return all(
            getattr(self, f).equal(getattr(o, f))
            for f in self.__annotations__
            if f != "prev_batched_graph"
        ) and (
            self.prev_batched_graph.batch.equal(o.prev_batched_graph.batch)
            and self.prev_batched_graph.x.equal(o.prev_batched_graph.x)
            and self.prev_batched_graph.node_label_mask.equal(
                o.prev_batched_graph.node_label_mask
            )
            and self.prev_batched_graph.node_last_update.equal(
                o.prev_batched_graph.node_last_update
            )
            and self.prev_batched_graph.edge_index.equal(
                o.prev_batched_graph.edge_index
            )
            and self.prev_batched_graph.edge_attr.equal(o.prev_batched_graph.edge_attr)
            and self.prev_batched_graph.edge_label_mask.equal(
                o.prev_batched_graph.edge_label_mask
            )
            and self.prev_batched_graph.edge_last_update.equal(
                o.prev_batched_graph.edge_last_update
            )
        )

    def to(self, *args, **kwargs) -> "TWCmdGenGraphEventGraphicalInput":
        return TWCmdGenGraphEventGraphicalInput(
            **{f: getattr(self, f).to(*args, **kwargs) for f in self.__annotations__}
        )

    def pin_memory(self) -> "TWCmdGenGraphEventGraphicalInput":
        return TWCmdGenGraphEventGraphicalInput(
            **{f: getattr(self, f).pin_memory() for f in self.__annotations__}
        )


@dataclass(frozen=True)
class TWCmdGenGraphEventBatch:
    ids: Tuple[Tuple[str, int, int], ...]
    step_input: TWCmdGenGraphEventStepInput
    initial_batched_graph: Batch
    graphical_input_seq: Tuple[TWCmdGenGraphEventGraphicalInput, ...]
    graph_commands: Tuple[Tuple[str, ...], ...]

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, TWCmdGenGraphEventBatch):
            return False
        return all(
            getattr(self, f) == getattr(o, f)
            for f in self.__annotations__
            if f != "initial_batched_graph"
        ) and (
            self.initial_batched_graph.batch.equal(o.initial_batched_graph.batch)
            and self.initial_batched_graph.x.equal(o.initial_batched_graph.x)
            and self.initial_batched_graph.node_label_mask.equal(
                o.initial_batched_graph.node_label_mask
            )
            and self.initial_batched_graph.node_last_update.equal(
                o.initial_batched_graph.node_last_update
            )
            and self.initial_batched_graph.edge_index.equal(
                o.initial_batched_graph.edge_index
            )
            and self.initial_batched_graph.edge_attr.equal(
                o.initial_batched_graph.edge_attr
            )
            and self.initial_batched_graph.edge_label_mask.equal(
                o.initial_batched_graph.edge_label_mask
            )
            and self.initial_batched_graph.edge_last_update.equal(
                o.initial_batched_graph.edge_last_update
            )
        )

    def to(self, *args, **kwargs) -> "TWCmdGenGraphEventBatch":
        return TWCmdGenGraphEventBatch(
            ids=self.ids,
            step_input=self.step_input.to(*args, **kwargs),
            initial_batched_graph=self.initial_batched_graph.to(*args, **kwargs),
            graphical_input_seq=tuple(
                graphical.to(*args, **kwargs) for graphical in self.graphical_input_seq
            ),
            graph_commands=self.graph_commands,
        )

    def pin_memory(self) -> "TWCmdGenGraphEventBatch":
        return TWCmdGenGraphEventBatch(
            ids=self.ids,
            step_input=self.step_input.pin_memory(),
            initial_batched_graph=self.initial_batched_graph.pin_memory(),
            graphical_input_seq=tuple(
                graphical.pin_memory() for graphical in self.graphical_input_seq
            ),
            graph_commands=self.graph_commands,
        )


def collate_step_inputs(
    preprocessor: Preprocessor,
    obs: List[str],
    prev_actions: List[str],
    timestamps: List[int],
) -> TWCmdGenGraphEventStepInput:
    """
    Collate step data such as observation, previous action and timestamp.

    output: TWCmdGenGraphEventStepInput(
        obs_word_ids: (batch, obs_len),
        obs_mask: (batch, obs_len),
        prev_action_word_ids: (batch, prev_action_len),
        prev_action_mask: (batch, prev_action_len),
        timestamps: (batch)
    )
    """
    # textual observation
    obs_word_ids, obs_mask = preprocessor.preprocess(obs)

    # textual previous action
    prev_action_word_ids, prev_action_mask = preprocessor.preprocess(prev_actions)
    return TWCmdGenGraphEventStepInput(
        obs_word_ids=obs_word_ids,
        obs_mask=obs_mask,
        prev_action_word_ids=prev_action_word_ids,
        prev_action_mask=prev_action_mask,
        timestamps=torch.tensor(timestamps),
    )


class TWCmdGenGraphEventDataCollator:
    def __init__(self, preprocessor: Preprocessor) -> None:
        self.preprocessor = preprocessor

    def collate_graphical_input_seq(
        self, batch: List[Dict[str, Any]], initial_batched_graph: Batch
    ) -> Tuple[TWCmdGenGraphEventGraphicalInput, ...]:
        """
        Collate the graphical input sequence of the given batch.

        output: len([TWCmdGenGraphEventGraphicalInput(
            tgt_event_type_ids: (batch),
            tgt_event_src_ids: (batch),
            tgt_event_dst_ids: (batch),
            tgt_event_label_word_ids: (batch, tgt_event_label_len),
            tgt_event_label_mask: (batch, tgt_event_label_len),
            groundtruth_event_type_ids: (batch),
            groundtruth_event_mask: (batch), boolean
            groundtruth_event_src_ids: (batch),
            groundtruth_event_src_mask: (batch), boolean
            groundtruth_event_dst_ids: (batch),
            groundtruth_event_dst_mask: (batch), boolean
            groundtruth_event_label_tgt_word_ids: (batch, groundtruth_event_label_len),
            groundtruth_event_label_tgt_mask:
                (batch, groundtruth_event_label_len), boolean
            groundtruth_event_label_groundtruth_word_ids:
                (batch, groundtruth_event_label_len),
            prev_batched_graph: Batch
        ), ...]) = event_seq_len
        """

        # collect current step's event sequences
        batch_event_seq: List[List[Dict[str, Any]]] = []
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
        tgt_event_label_word_ids = torch.empty(batch_size, 0, dtype=torch.long)
        tgt_event_label_mask = torch.empty(batch_size, 0, dtype=torch.bool)
        tgt_event_timestamps = torch.tensor([step["timestamp"] for step in batch])

        prev_batched_graph = initial_batched_graph
        collated: List[TWCmdGenGraphEventGraphicalInput] = []
        for seq_step_num in range(max_event_seq_len):
            batch_event_type_ids: List[int] = []
            batch_event_src_ids: List[int] = []
            batch_event_dst_ids: List[int] = []
            batch_event_groundtruth_labels: List[str] = []
            for event_seq in batch_event_seq:
                # collect event data for all the items in the batch at event i
                if seq_step_num < len(event_seq):
                    event = event_seq[seq_step_num]
                    batch_event_type_ids.append(EVENT_TYPE_ID_MAP[event["type"]])
                    batch_event_src_ids.append(
                        event.get("src_id", event.get("node_id", 0))
                    )
                    batch_event_dst_ids.append(event.get("dst_id", 0))
                    batch_event_groundtruth_labels.append(event["label"])
                else:
                    batch_event_type_ids.append(EVENT_TYPE_ID_MAP["pad"])
                    batch_event_src_ids.append(0)
                    batch_event_dst_ids.append(0)
                    # can't pack an empty sequence so add an empty string
                    batch_event_groundtruth_labels.append("")
            groundtruth_event_type_ids = torch.tensor(batch_event_type_ids)
            groundtruth_event_src_ids = torch.tensor(batch_event_src_ids)
            groundtruth_event_dst_ids = torch.tensor(batch_event_dst_ids)
            (
                groundtruth_event_label_groundtruth_word_ids,
                groundtruth_event_label_tgt_mask,
            ) = self.preprocessor.preprocess(batch_event_groundtruth_labels)
            groundtruth_event_label_tgt_word_ids = shift_tokens_right(
                groundtruth_event_label_groundtruth_word_ids,
                self.preprocessor.bos_token_id,
            )
            masks = compute_masks_from_event_type_ids(groundtruth_event_type_ids)
            collated.append(
                TWCmdGenGraphEventGraphicalInput(
                    tgt_event_type_ids=tgt_event_type_ids,
                    tgt_event_src_ids=tgt_event_src_ids,
                    tgt_event_dst_ids=tgt_event_dst_ids,
                    tgt_event_label_word_ids=tgt_event_label_word_ids,
                    tgt_event_label_mask=tgt_event_label_mask,
                    groundtruth_event_type_ids=groundtruth_event_type_ids,
                    groundtruth_event_mask=masks["event_mask"],
                    groundtruth_event_src_ids=groundtruth_event_src_ids,
                    groundtruth_event_src_mask=masks["src_mask"],
                    groundtruth_event_dst_ids=groundtruth_event_dst_ids,
                    groundtruth_event_dst_mask=masks["dst_mask"],
                    groundtruth_event_label_tgt_word_ids=(
                        groundtruth_event_label_tgt_word_ids
                    ),
                    groundtruth_event_label_tgt_mask=groundtruth_event_label_tgt_mask,
                    groundtruth_event_label_groundtruth_word_ids=(
                        groundtruth_event_label_groundtruth_word_ids
                    ),
                    groundtruth_event_label_mask=masks["label_mask"],
                    prev_batched_graph=prev_batched_graph,
                )
            )

            # update the previous batched graph
            prev_batched_graph = update_batched_graph(
                prev_batched_graph,
                tgt_event_type_ids,
                tgt_event_src_ids,
                tgt_event_dst_ids,
                tgt_event_label_word_ids,
                tgt_event_label_mask,
                torch.stack(
                    [
                        tgt_event_timestamps,
                        # we subtract 1 here since the target sequence is right-shfited
                        torch.tensor(seq_step_num - 1).expand(batch_size),
                    ],
                    dim=1,
                ),
            )

            # the current groundtruth events become the next target events
            tgt_event_type_ids = groundtruth_event_type_ids
            tgt_event_src_ids = groundtruth_event_src_ids
            tgt_event_dst_ids = groundtruth_event_dst_ids
            tgt_event_label_word_ids = groundtruth_event_label_groundtruth_word_ids
            tgt_event_label_mask = groundtruth_event_label_tgt_mask

        return tuple(collated)

    def collate_prev_graph_events(self, batch: List[Dict[str, Any]]) -> Batch:
        """
        Collate the previous graph events of the given batch into a single
        batched global graph.

        output: Batch(
            batch: (num_node)
            x: (num_node, node_label_len)
            node_label_mask: (num_node, node_label_len)
            node_last_update: (num_node, 2)
            edge_index: (2, num_edge)
            edge_attr: (num_edge, edge_label_len)
            edge_label_mask: (num_edge, edge_label_len)
            edge_last_update: (num_edge, 2)
        )
        """
        batch_prev_event_seq = [step["prev_graph_events"] for step in batch]
        # initialize empty batched graph
        batched_graph = empty_graph()

        max_prev_event_seq_len = max(
            len(event_seq) for event_seq in batch_prev_event_seq
        )

        for seq_step_num in range(max_prev_event_seq_len):
            batch_event_type_ids: List[int] = []
            batch_event_src_ids: List[int] = []
            batch_event_dst_ids: List[int] = []
            batch_event_labels: List[str] = []
            batch_event_timestamps: List[List[int]] = []
            for event_seq in batch_prev_event_seq:
                # collect event data for all the items in the batch at event i
                if seq_step_num < len(event_seq):
                    event = event_seq[seq_step_num]
                    batch_event_type_ids.append(EVENT_TYPE_ID_MAP[event["type"]])
                    batch_event_src_ids.append(
                        event.get("src_id", event.get("node_id", 0))
                    )
                    batch_event_dst_ids.append(event.get("dst_id", 0))
                    batch_event_labels.append(event["label"])
                    batch_event_timestamps.append(event["timestamp"])
                else:
                    batch_event_type_ids.append(EVENT_TYPE_ID_MAP["pad"])
                    batch_event_src_ids.append(0)
                    batch_event_dst_ids.append(0)
                    batch_event_labels.append("")
                    batch_event_timestamps.append([0, 0])
            event_label_word_ids, event_label_mask = self.preprocessor.preprocess(
                batch_event_labels
            )
            batched_graph = update_batched_graph(
                batched_graph,
                torch.tensor(batch_event_type_ids),
                torch.tensor(batch_event_src_ids),
                torch.tensor(batch_event_dst_ids),
                event_label_word_ids,
                event_label_mask,
                torch.tensor(batch_event_timestamps),
            )

        return batched_graph

    def __call__(self, batch: List[Dict[str, Any]]) -> TWCmdGenGraphEventBatch:
        """
        Each batch has a batch of IDs, batched textual input, batched graphical
        input, batched previous graph events and ground-truth graph commands.

        TWCmdGenGraphEventBatch(
            ids=(
                (game, walkthrough_step, random_step),
                ...
            ),
            step_input=TWCmdGenGraphEventStepInput(
                obs_word_ids: (batch, obs_len),
                obs_mask: (batch, obs_len),
                prev_action_word_ids: (batch, prev_action_len),
                prev_action_mask: (batch, prev_action_len),
                timestamps: (batch)
            ),
            graphical_input_seq=(
                TWCmdGenGraphEventGraphicalInput(
                    tgt_event_type_ids: (batch),
                    tgt_event_src_ids: (batch),
                    tgt_event_dst_ids: (batch),
                    tgt_event_label_ids: (batch),
                    prev_batched_graph: Batch,
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
            graph_commands=(
                (
                    (commands, ...),
                    ...
                ),
                ...
            )
        )
        """
        initial_batched_graph = self.collate_prev_graph_events(batch)
        return TWCmdGenGraphEventBatch(
            ids=tuple(
                (
                    str(step["game"]),
                    int(step["walkthrough_step"]),
                    int(step["random_step"]),
                )
                for step in batch
            ),
            step_input=collate_step_inputs(
                self.preprocessor,
                [step["observation"] for step in batch],
                [step["previous_action"] for step in batch],
                [step["timestamp"] for step in batch],
            ),
            initial_batched_graph=initial_batched_graph,
            graphical_input_seq=self.collate_graphical_input_seq(
                batch, initial_batched_graph
            ),
            graph_commands=tuple(tuple(step["target_commands"]) for step in batch),
        )


@dataclass(frozen=True)
class TWCmdGenObsGenBatch:
    ids: Tuple[Tuple[str, int], ...]
    step_inputs: Tuple[TWCmdGenGraphEventStepInput, ...]
    step_mask: torch.Tensor

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, TWCmdGenObsGenBatch):
            return False
        return all(
            getattr(self, f) == getattr(o, f)
            for f in self.__annotations__
            if f != "step_mask"
        ) and self.step_mask.equal(o.step_mask)

    def to(self, *args, **kwargs) -> "TWCmdGenObsGenBatch":
        return TWCmdGenObsGenBatch(
            ids=self.ids,
            step_inputs=tuple(
                step_input.to(*args, **kwargs) for step_input in self.step_inputs
            ),
            step_mask=self.step_mask.to(*args, **kwargs),
        )

    def pin_memory(self) -> "TWCmdGenObsGenBatch":
        return TWCmdGenObsGenBatch(
            ids=self.ids,
            step_inputs=tuple(
                step_input.pin_memory() for step_input in self.step_inputs
            ),
            step_mask=self.step_mask.pin_memory(),
        )


class TWCmdGenObsGenDataCollator:
    def __init__(self, preprocessor: Preprocessor) -> None:
        self.preprocessor = preprocessor

    def __call__(self, batch: List[Dict[str, Any]]) -> TWCmdGenObsGenBatch:
        """
        Each batch is a tuple of step inputs and a step mask.
        (
            ids: ((game, walkthrough_step), ...)
            step_inputs: (TWCmdGenGraphEventStepInput(
                obs_word_ids: (batch, obs_len),
                obs_mask: (batch, obs_len),
                prev_action_word_ids: (batch, prev_action_len),
                prev_action_mask: (batch, prev_action_len),
                timestamps: (batch)
            ), ...),
            step_mask: (max_step, batch)
        )
        """
        max_step = max(len(walkthrough["steps"]) for walkthrough in batch)
        collated_steps: List[TWCmdGenGraphEventStepInput] = []
        collated_step_mask: List[List[bool]] = []
        for step in range(max_step):
            collated_steps.append(
                collate_step_inputs(
                    self.preprocessor,
                    [
                        walkthrough["steps"][step]["observation"]
                        if step < len(walkthrough["steps"])
                        else ""
                        for walkthrough in batch
                    ],
                    [
                        walkthrough["steps"][step]["previous_action"]
                        if step < len(walkthrough["steps"])
                        else ""
                        for walkthrough in batch
                    ],
                    [
                        step if step < len(walkthrough["steps"]) else 0
                        for walkthrough in batch
                    ],
                )
            )
            collated_step_mask.append(
                [
                    True if step < len(walkthrough["steps"]) else False
                    for walkthrough in batch
                ]
            )
        return TWCmdGenObsGenBatch(
            ids=tuple(
                (walkthrough["game"], walkthrough["walkthrough_step"])
                for walkthrough in batch
            ),
            step_inputs=tuple(collated_steps),
            step_mask=torch.tensor(collated_step_mask),
        )


class TWCmdGenGraphEventDataModule(pl.LightningDataModule):
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
        preprocessor: Preprocessor,
        allow_objs_with_same_label: bool = False,
        sort_commands: bool = False,
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
        self.allow_objs_with_same_label = allow_objs_with_same_label
        self.sort_commands = sort_commands

        self.preprocessor = preprocessor
        self.collator = TWCmdGenGraphEventDataCollator(self.preprocessor)

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train = TWCmdGenGraphEventDataset(
                self.train_path,
                allow_objs_with_same_label=self.allow_objs_with_same_label,
                sort_commands=self.sort_commands,
            )
            self.valid = TWCmdGenGraphEventDataset(
                self.val_path,
                allow_objs_with_same_label=self.allow_objs_with_same_label,
                sort_commands=self.sort_commands,
            )
            self.valid_free_run = TWCmdGenGraphEventFreeRunDataset(
                self.val_path,
                self.val_batch_size,
                allow_objs_with_same_label=self.allow_objs_with_same_label,
                sort_commands=self.sort_commands,
            )

        if stage == "test" or stage is None:
            self.test = TWCmdGenGraphEventDataset(
                self.test_path,
                allow_objs_with_same_label=self.allow_objs_with_same_label,
                sort_commands=self.sort_commands,
            )
            self.test_free_run = TWCmdGenGraphEventFreeRunDataset(
                self.test_path,
                self.test_batch_size,
                allow_objs_with_same_label=self.allow_objs_with_same_label,
                sort_commands=self.sort_commands,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.train_batch_size,
            collate_fn=self.collator,
            shuffle=True,
            pin_memory=True,
            num_workers=self.train_num_worker,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid,
            batch_size=self.val_batch_size,
            collate_fn=self.collator,
            pin_memory=True,
            num_workers=self.val_num_worker,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            batch_size=self.test_batch_size,
            collate_fn=self.collator,
            pin_memory=True,
            num_workers=self.test_num_worker,
        )

    def val_free_run_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_free_run, batch_size=None, pin_memory=True, num_workers=1
        )

    def test_free_run_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_free_run, batch_size=None, pin_memory=True, num_workers=1
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


class TWCmdGenObsGenDataModule(pl.LightningDataModule):
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
        preprocessor: Preprocessor,
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

        self.preprocessor = preprocessor
        self.collator = TWCmdGenObsGenDataCollator(preprocessor)

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train = TWCmdGenObsGenDataset(self.train_path)
            self.valid = TWCmdGenObsGenDataset(self.val_path)

        if stage == "test" or stage is None:
            self.test = TWCmdGenObsGenDataset(self.test_path)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.train_batch_size,
            collate_fn=self.collator,
            shuffle=True,
            pin_memory=True,
            num_workers=self.train_num_worker,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid,
            batch_size=self.val_batch_size,
            collate_fn=self.collator,
            pin_memory=True,
            num_workers=self.val_num_worker,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            batch_size=self.test_batch_size,
            collate_fn=self.collator,
            pin_memory=True,
            num_workers=self.test_num_worker,
        )
