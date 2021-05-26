import pytest

from dgu.data import TemporalDataBatchSampler, TWCmdGenDataset
from dgu.graph import TextWorldGraph


@pytest.mark.parametrize(
    "batch_size,event_seq_lens,expected_batches",
    [
        (1, [1], [[0]]),
        (
            3,
            [1, 3, 6, 5],
            [[0], [1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14]],
        ),
    ],
)
def test_temporal_data_batch_sampler(batch_size, event_seq_lens, expected_batches):
    sampler = TemporalDataBatchSampler(batch_size, event_seq_lens)
    assert list(iter(sampler)) == expected_batches
    assert len(sampler) == len(expected_batches)


@pytest.mark.parametrize(
    "cmds,steps,expected_events,expected_node_labels,expected_edge_labels",
    [
        (
            ["add , player , kitchen , in"],
            [0, 0],
            [
                {"type": "node-add", "node_id": 0, "timestamp": 0},
                {"type": "node-add", "node_id": 1, "timestamp": 0},
                {"type": "edge-add", "src_id": 0, "dst_id": 1, "timestamp": 0},
            ],
            [],
            [],
        )
    ],
)
def test_tw_cmd_gen_dataset_transform_cmd_events(
    cmds, steps, expected_events, expected_node_labels, expected_edge_labels
):
    g = TextWorldGraph()
    assert (
        TWCmdGenDataset.transform_commands_to_events(cmds, steps, g) == expected_events
    )
