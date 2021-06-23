import pytest
import torch
import torch.nn as nn

from dgu.nn.temporal_graph import TemporalGraphNetwork
from dgu.nn.utils import compute_masks_from_event_type_ids
from dgu.constants import EVENT_TYPE_ID_MAP


@pytest.mark.parametrize(
    "num_nodes,event_type_ids,src_ids,dst_ids,event_embeddings,event_timestamps,"
    "src_expected,dst_expected",
    [
        (
            5,
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["end"],
                ]
            ),
            torch.tensor([1, 1, 3]),
            torch.tensor([1, 1, 2]),
            torch.tensor([[1] * 4, [2] * 4, [3] * 4]),
            torch.tensor([1, 2, 3]),
            torch.tensor(
                [
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [3] * 4 + [0] * 4 + [0] * 4 + [5] * 4 + [2] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                ]
            ).float(),
            torch.zeros(3, 20),
        ),
        (
            10,
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["node-delete"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                    EVENT_TYPE_ID_MAP["end"],
                    EVENT_TYPE_ID_MAP["pad"],
                    EVENT_TYPE_ID_MAP["pad"],
                ]
            ),
            torch.tensor([0, 0, 1, 2, 3, 0, 0, 0]),
            torch.tensor([0, 0, 0, 2, 1, 0, 0, 0]),
            torch.tensor([[i + 1] * 4 for i in range(8)]),
            torch.linspace(1, 9, 8).long(),
            torch.tensor(
                [
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [3] * 4 + [0] * 4 + [0] * 4 + [5] * 4 + [2] * 4,
                    [4] * 4 + [1] * 4 + [0] * 4 + [6] * 4 + [3] * 4,
                    [5] * 4 + [2] * 4 + [2] * 4 + [3] * 4 + [4] * 4,
                    [6] * 4 + [3] * 4 + [1] * 4 + [2] * 4 + [5] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                ]
            ).float(),
            torch.tensor(
                [
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [5] * 4 + [2] * 4 + [2] * 4 + [3] * 4 + [4] * 4,
                    [6] * 4 + [1] * 4 + [3] * 4 + [6] * 4 + [5] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                ]
            ).float(),
        ),
    ],
)
def test_tgn_message(
    num_nodes,
    event_type_ids,
    src_ids,
    dst_ids,
    event_embeddings,
    event_timestamps,
    src_expected,
    dst_expected,
):
    tgn = TemporalGraphNetwork(4)
    for i in range(num_nodes):
        tgn.memory[i] = torch.tensor([i] * 4).float()
        tgn.last_update[i] = i * 2

    class MockTimeEncoder(nn.Module):
        def forward(self, timestamp):
            return timestamp.unsqueeze(-1).expand(-1, 4) + 3

    tgn.time_encoder = MockTimeEncoder()

    event_mask, src_mask, dst_mask = compute_masks_from_event_type_ids(event_type_ids)
    src_messages, dst_messages = tgn.message(
        event_type_ids,
        src_ids,
        src_mask,
        dst_ids,
        dst_mask,
        event_embeddings,
        event_mask,
        event_timestamps,
    )
    assert src_messages.equal(src_expected)
    assert dst_messages.equal(dst_expected)


@pytest.mark.parametrize(
    "messages,ids,expected",
    [
        (torch.ones(1, 4), torch.tensor([0]), torch.ones(1, 4)),
        (
            torch.cat([torch.ones(1, 4), torch.zeros(1, 4), torch.ones(1, 4)]),
            torch.tensor([0, 1, 1]),
            torch.cat([torch.ones(1, 4), torch.tensor([[0.5, 0.5, 0.5, 0.5]])]),
        ),
    ],
)
def test_tgn_agg_message(messages, ids, expected):
    tgn = TemporalGraphNetwork(4)
    assert tgn.agg_message(messages, ids).equal(expected)
