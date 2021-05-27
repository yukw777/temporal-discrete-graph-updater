import itertools
import json

from tqdm import tqdm
from typing import List, Iterator, Dict, Any
from torch.utils.data import Sampler, Dataset

from dgu.graph import TextWorldGraph


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
        }
    Each edge event contains the following information:
        {
            "type": "edge-{add,delete}",
            "src_id": id for src node to be added/deleted,
            "dst_id": id for dst node to be added/deleted,
            "timestamp": timestamp for the event,
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
