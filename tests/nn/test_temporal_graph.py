import pytest
import torch
import torch.nn as nn

from dgu.nn.temporal_graph import TemporalGraphNetwork, TransformerConvStack
from dgu.constants import EVENT_TYPES, EVENT_TYPE_ID_MAP


class MockTimeEncoder(nn.Module):
    def forward(self, timestamp):
        return timestamp.unsqueeze(-1).expand(-1, 4) + 3


@pytest.mark.parametrize(
    "event_type_ids,event_type_emb,node_ids,event_embeddings,event_timestamps,"
    "event_mask,memory,expected",
    [
        (
            torch.tensor([EVENT_TYPE_ID_MAP["node-add"]]),
            torch.linspace(0, 6, 7).unsqueeze(-1).expand(-1, 4),
            torch.tensor([0]),
            torch.tensor([[3] * 4]).float(),
            torch.tensor([0]),
            torch.tensor([1]).float(),
            torch.linspace(4, 7, 4).unsqueeze(-1).expand(-1, 4),
            torch.tensor(
                [
                    [3] * 4 + [4] * 4 + [0] * 4 + [3] * 4 + [3] * 4,
                ]
            ).float(),
        ),
        (
            torch.tensor([EVENT_TYPE_ID_MAP["node-delete"]]),
            torch.linspace(0, 6, 7).unsqueeze(-1).expand(-1, 4),
            torch.tensor([0]),
            torch.tensor([[3] * 4]).float(),
            torch.tensor([0]),
            torch.tensor([1]).float(),
            torch.linspace(4, 7, 4).unsqueeze(-1).expand(-1, 4),
            torch.tensor(
                [
                    [4] * 4 + [4] * 4 + [0] * 4 + [3] * 4 + [3] * 4,
                ]
            ).float(),
        ),
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["end"],
                ]
            ),
            torch.linspace(1, 7, 7).unsqueeze(-1).expand(-1, 4),
            torch.tensor([0, 3, 0]),
            torch.tensor([[0] * 4, [8] * 4, [0] * 4]).float(),
            torch.tensor([0, 1, 0]),
            torch.tensor([0, 1, 0]).float(),
            torch.linspace(9, 12, 4).unsqueeze(-1).expand(-1, 4),
            torch.tensor(
                [
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [4] * 4 + [12] * 4 + [0] * 4 + [4] * 4 + [8] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                ]
            ).float(),
        ),
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["node-delete"],
                    EVENT_TYPE_ID_MAP["end"],
                    EVENT_TYPE_ID_MAP["pad"],
                ]
            ),
            torch.linspace(1, 7, 7).unsqueeze(-1).expand(-1, 4),
            torch.tensor([0, 3, 0, 0]),
            torch.tensor([[0] * 4, [8] * 4, [0] * 4, [0] * 4]).float(),
            torch.tensor([0, 1, 0, 0]),
            torch.tensor([0, 1, 0, 0]).float(),
            torch.linspace(9, 12, 4).unsqueeze(-1).expand(-1, 4),
            torch.tensor(
                [
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [5] * 4 + [12] * 4 + [0] * 4 + [4] * 4 + [8] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                ]
            ).float(),
        ),
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["node-delete"],
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["end"],
                ]
            ),
            torch.linspace(1, 7, 7).unsqueeze(-1).expand(-1, 4),
            torch.tensor([0, 1, 2, 3, 0]),
            torch.tensor([[0] * 4, [8] * 4, [9] * 4, [10] * 4, [0] * 4]).float(),
            torch.tensor([0, 1, 2, 3, 0]),
            torch.tensor([0, 1, 1, 1, 0]).float(),
            torch.linspace(11, 14, 4).unsqueeze(-1).expand(-1, 4),
            torch.tensor(
                [
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [4] * 4 + [12] * 4 + [0] * 4 + [4] * 4 + [8] * 4,
                    [5] * 4 + [13] * 4 + [0] * 4 + [5] * 4 + [9] * 4,
                    [4] * 4 + [14] * 4 + [0] * 4 + [6] * 4 + [10] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                ]
            ).float(),
        ),
    ],
)
def test_tgn_node_message(
    event_type_ids,
    event_type_emb,
    node_ids,
    event_embeddings,
    event_timestamps,
    event_mask,
    memory,
    expected,
):
    tgn = TemporalGraphNetwork(0, 0, 4, 4, 4, 4, 8, 1, 1)
    tgn.time_encoder = MockTimeEncoder()
    tgn.event_type_emb = nn.Embedding.from_pretrained(event_type_emb)

    assert tgn.node_message(
        event_type_ids, node_ids, event_embeddings, event_timestamps, event_mask, memory
    ).equal(expected)


@pytest.mark.parametrize(
    "event_type_ids,event_type_emb,src_ids,dst_ids,event_edge_ids,event_embeddings,"
    "event_timestamps,event_mask,memory,last_update,src_expected,dst_expected",
    [
        (
            torch.tensor([EVENT_TYPE_ID_MAP["edge-add"]]),
            torch.linspace(0, 6, 7).unsqueeze(-1).expand(-1, 4),
            torch.tensor([0]),
            torch.tensor([1]),
            torch.tensor([0]),
            torch.tensor([[3] * 4]).float(),
            torch.tensor([2]),
            torch.tensor([1]).float(),
            torch.linspace(4, 7, 4).unsqueeze(-1).expand(-1, 4),
            torch.tensor([1, 2, 3]),
            torch.tensor(
                [
                    [5] * 4 + [4] * 4 + [5] * 4 + [4] * 4 + [3] * 4,
                ]
            ).float(),
            torch.tensor(
                [
                    [5] * 4 + [5] * 4 + [4] * 4 + [4] * 4 + [3] * 4,
                ]
            ).float(),
        ),
        (
            torch.tensor([EVENT_TYPE_ID_MAP["edge-delete"]]),
            torch.linspace(0, 6, 7).unsqueeze(-1).expand(-1, 4),
            torch.tensor([0]),
            torch.tensor([1]),
            torch.tensor([0]),
            torch.tensor([[3] * 4]).float(),
            torch.tensor([2]),
            torch.tensor([1]).float(),
            torch.linspace(4, 7, 4).unsqueeze(-1).expand(-1, 4),
            torch.tensor([1, 2, 3]),
            torch.tensor(
                [
                    [6] * 4 + [4] * 4 + [5] * 4 + [4] * 4 + [3] * 4,
                ]
            ).float(),
            torch.tensor(
                [
                    [6] * 4 + [5] * 4 + [4] * 4 + [4] * 4 + [3] * 4,
                ]
            ).float(),
        ),
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["end"],
                ]
            ),
            torch.linspace(1, 7, 7).unsqueeze(-1).expand(-1, 4),
            torch.tensor([0, 2, 0]),
            torch.tensor([0, 3, 0]),
            torch.tensor([0, 4, 0]),
            torch.tensor([[0] * 4, [8] * 4, [0] * 4]).float(),
            torch.tensor([0, 4, 0]),
            torch.tensor([0, 1, 0]).float(),
            torch.linspace(9, 12, 4).unsqueeze(-1).expand(-1, 4),
            torch.tensor([3, 2, 4, 5, 3]),
            torch.tensor(
                [
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [6] * 4 + [11] * 4 + [12] * 4 + [4] * 4 + [8] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                ]
            ).float(),
            torch.tensor(
                [
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [6] * 4 + [12] * 4 + [11] * 4 + [4] * 4 + [8] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                ]
            ).float(),
        ),
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                    EVENT_TYPE_ID_MAP["end"],
                ]
            ),
            torch.linspace(1, 7, 7).unsqueeze(-1).expand(-1, 4),
            torch.tensor([0, 2, 0]),
            torch.tensor([0, 3, 0]),
            torch.tensor([0, 4, 0]),
            torch.tensor([[0] * 4, [8] * 4, [0] * 4]).float(),
            torch.tensor([0, 4, 0]),
            torch.tensor([0, 1, 0]).float(),
            torch.linspace(9, 12, 4).unsqueeze(-1).expand(-1, 4),
            torch.tensor([3, 2, 4, 5, 3]),
            torch.tensor(
                [
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [7] * 4 + [11] * 4 + [12] * 4 + [4] * 4 + [8] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                ]
            ).float(),
            torch.tensor(
                [
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [7] * 4 + [12] * 4 + [11] * 4 + [4] * 4 + [8] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                ]
            ).float(),
        ),
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                    EVENT_TYPE_ID_MAP["end"],
                    EVENT_TYPE_ID_MAP["pad"],
                ]
            ),
            torch.linspace(1, 7, 7).unsqueeze(-1).expand(-1, 4),
            torch.tensor([0, 1, 0, 0]),
            torch.tensor([0, 2, 0, 0]),
            torch.tensor([0, 3, 0, 0]),
            torch.tensor([[0] * 4, [8] * 4, [0] * 4, [0] * 4]).float(),
            torch.tensor([0, 5, 0, 0]),
            torch.tensor([0, 1, 0, 0]).float(),
            torch.linspace(9, 12, 4).unsqueeze(-1).expand(-1, 4),
            torch.tensor([3, 2, 4, 5, 3]),
            torch.tensor(
                [
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [7] * 4 + [10] * 4 + [11] * 4 + [3] * 4 + [8] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                ]
            ).float(),
            torch.tensor(
                [
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [7] * 4 + [11] * 4 + [10] * 4 + [3] * 4 + [8] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                ]
            ).float(),
        ),
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["end"],
                ]
            ),
            torch.linspace(1, 7, 7).unsqueeze(-1).expand(-1, 4),
            torch.tensor([0, 1, 2, 3, 0]),
            torch.tensor([0, 3, 2, 1, 0]),
            torch.tensor([0, 2, 3, 4, 0]),
            torch.tensor([[0] * 4, [8] * 4, [9] * 4, [10] * 4, [0] * 4]).float(),
            torch.tensor([0, 4, 5, 6, 0]),
            torch.tensor([0, 1, 1, 1, 0]).float(),
            torch.linspace(11, 14, 4).unsqueeze(-1).expand(-1, 4),
            torch.tensor([3, 2, 4, 5, 3]),
            torch.tensor(
                [
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [6] * 4 + [12] * 4 + [14] * 4 + [3] * 4 + [8] * 4,
                    [7] * 4 + [13] * 4 + [13] * 4 + [3] * 4 + [9] * 4,
                    [6] * 4 + [14] * 4 + [12] * 4 + [6] * 4 + [10] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                ]
            ).float(),
            torch.tensor(
                [
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                    [6] * 4 + [14] * 4 + [12] * 4 + [3] * 4 + [8] * 4,
                    [7] * 4 + [13] * 4 + [13] * 4 + [3] * 4 + [9] * 4,
                    [6] * 4 + [12] * 4 + [14] * 4 + [6] * 4 + [10] * 4,
                    [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4 + [0] * 4,
                ]
            ).float(),
        ),
    ],
)
def test_tgn_edge_message(
    event_type_ids,
    event_type_emb,
    src_ids,
    dst_ids,
    event_edge_ids,
    event_embeddings,
    event_timestamps,
    event_mask,
    memory,
    last_update,
    src_expected,
    dst_expected,
):
    tgn = TemporalGraphNetwork(0, 0, 4, 4, 4, 4, 8, 1, 1)
    tgn.time_encoder = MockTimeEncoder()
    tgn.event_type_emb = nn.Embedding.from_pretrained(event_type_emb)

    src_messages, dst_messages = tgn.edge_message(
        event_type_ids,
        src_ids,
        dst_ids,
        event_edge_ids,
        event_embeddings,
        event_timestamps,
        event_mask,
        memory,
        last_update,
    )
    assert src_messages.equal(src_expected)
    assert dst_messages.equal(dst_expected)


@pytest.mark.parametrize(
    "messages,ids,expected",
    [
        (torch.ones(1, 1, 4), torch.tensor([[0]]), torch.ones(1, 4)),
        (
            torch.cat(
                [torch.ones(1, 4), torch.zeros(1, 4), torch.ones(1, 4)]
            ).unsqueeze(0),
            torch.tensor([[0, 1, 1]]),
            torch.ones(2, 4),
        ),
        (
            torch.stack(
                [
                    torch.cat([torch.ones(1, 4), torch.zeros(1, 4), torch.ones(1, 4)]),
                    torch.cat([torch.zeros(1, 4), torch.zeros(1, 4), torch.ones(1, 4)]),
                ]
            ),
            torch.tensor([[0, 1, 1], [0, 1, 0]]),
            torch.tensor([[2.0, 2.0, 2.0, 2.0], [1.0, 1.0, 1.0, 1.0]]),
        ),
    ],
)
def test_tgn_agg_message(messages, ids, expected):
    tgn = TemporalGraphNetwork(10, 10, 4, 4, 4, 4, 8, 1, 1)
    assert tgn.agg_message(messages, ids).equal(expected)


@pytest.mark.parametrize(
    "event_type_ids,src_ids,event_embeddings,expected",
    [
        (
            torch.tensor(
                [
                    [
                        EVENT_TYPE_ID_MAP["node-delete"],
                        EVENT_TYPE_ID_MAP["edge-add"],
                        EVENT_TYPE_ID_MAP["edge-delete"],
                    ]
                ]
            ),
            torch.tensor([[1, 2, 3]]).long(),
            torch.tensor([[0.0, 1.0, 2.0]]).unsqueeze(-1).expand(-1, -1, 16),
            torch.zeros(5, 16),
        ),
        (
            torch.tensor(
                [
                    [
                        EVENT_TYPE_ID_MAP["start"],
                        EVENT_TYPE_ID_MAP["node-add"],
                        EVENT_TYPE_ID_MAP["end"],
                    ]
                ]
            ),
            torch.tensor([[0, 1, 0]]).long(),
            torch.tensor([[0.0, 1.0, 2.0]]).unsqueeze(-1).expand(-1, -1, 16),
            torch.tensor([[0] * 16, [1] * 16, [0] * 16, [0] * 16, [0] * 16]).float(),
        ),
        (
            torch.tensor(
                [
                    [
                        EVENT_TYPE_ID_MAP["node-add"],
                        EVENT_TYPE_ID_MAP["node-add"],
                    ]
                ]
            ),
            torch.tensor([[1, 1]]).long(),
            torch.tensor([[0.0, 1.0]]).unsqueeze(-1).expand(-1, -1, 16),
            torch.tensor([[0] * 16, [1] * 16, [0] * 16, [0] * 16, [0] * 16]).float(),
        ),
        (
            torch.tensor(
                [
                    [
                        EVENT_TYPE_ID_MAP["start"],
                        EVENT_TYPE_ID_MAP["node-add"],
                        EVENT_TYPE_ID_MAP["node-delete"],
                        EVENT_TYPE_ID_MAP["edge-add"],
                        EVENT_TYPE_ID_MAP["edge-delete"],
                        EVENT_TYPE_ID_MAP["end"],
                        EVENT_TYPE_ID_MAP["pad"],
                        EVENT_TYPE_ID_MAP["pad"],
                    ],
                    [
                        EVENT_TYPE_ID_MAP["start"],
                        EVENT_TYPE_ID_MAP["node-add"],
                        EVENT_TYPE_ID_MAP["node-delete"],
                        EVENT_TYPE_ID_MAP["edge-add"],
                        EVENT_TYPE_ID_MAP["edge-delete"],
                        EVENT_TYPE_ID_MAP["node-add"],
                        EVENT_TYPE_ID_MAP["edge-delete"],
                        EVENT_TYPE_ID_MAP["end"],
                    ],
                ]
            ),
            torch.tensor([[0, 0, 1, 2, 2, 0, 0, 0], [0, 3, 4, 3, 4, 4, 3, 0]]),
            torch.tensor([list(range(8)), [i + 2 for i in range(8)]])
            .float()
            .unsqueeze(-1)
            .expand(-1, -1, 16),
            torch.tensor([[1] * 16, [0] * 16, [0] * 16, [3] * 16, [7] * 16]).float(),
        ),
    ],
)
def test_tgn_update_node_features(event_type_ids, src_ids, event_embeddings, expected):
    tgn = TemporalGraphNetwork(5, 5, 10, 10, 10, 16, 10, 1, 1)
    tgn.update_node_features(event_type_ids, src_ids, event_embeddings)
    assert tgn.node_features.equal(expected)


@pytest.mark.parametrize(
    "event_type_ids,event_edge_ids,event_embeddings,expected",
    [
        (
            torch.tensor(
                [
                    [
                        EVENT_TYPE_ID_MAP["node-add"],
                        EVENT_TYPE_ID_MAP["node-delete"],
                        EVENT_TYPE_ID_MAP["edge-delete"],
                    ]
                ]
            ),
            torch.tensor([[1, 2, 3]]).long(),
            torch.tensor([[0.0, 1.0, 2.0]]).unsqueeze(-1).expand(-1, -1, 16),
            torch.zeros(5, 16),
        ),
        (
            torch.tensor(
                [
                    [
                        EVENT_TYPE_ID_MAP["start"],
                        EVENT_TYPE_ID_MAP["edge-add"],
                        EVENT_TYPE_ID_MAP["end"],
                    ]
                ]
            ),
            torch.tensor([[0, 1, 0]]).long(),
            torch.tensor([[0.0, 1.0, 2.0]]).unsqueeze(-1).expand(-1, -1, 16),
            torch.tensor([[0] * 16, [1] * 16, [0] * 16, [0] * 16, [0] * 16]).float(),
        ),
        (
            torch.tensor(
                [
                    [
                        EVENT_TYPE_ID_MAP["edge-add"],
                        EVENT_TYPE_ID_MAP["edge-add"],
                    ]
                ]
            ),
            torch.tensor([[1, 1]]).long(),
            torch.tensor([[0.0, 1.0]]).unsqueeze(-1).expand(-1, -1, 16),
            torch.tensor([[0] * 16, [1] * 16, [0] * 16, [0] * 16, [0] * 16]).float(),
        ),
        (
            torch.tensor(
                [
                    [
                        EVENT_TYPE_ID_MAP["start"],
                        EVENT_TYPE_ID_MAP["node-add"],
                        EVENT_TYPE_ID_MAP["node-delete"],
                        EVENT_TYPE_ID_MAP["edge-add"],
                        EVENT_TYPE_ID_MAP["edge-delete"],
                        EVENT_TYPE_ID_MAP["end"],
                        EVENT_TYPE_ID_MAP["pad"],
                        EVENT_TYPE_ID_MAP["pad"],
                    ],
                    [
                        EVENT_TYPE_ID_MAP["start"],
                        EVENT_TYPE_ID_MAP["node-add"],
                        EVENT_TYPE_ID_MAP["node-delete"],
                        EVENT_TYPE_ID_MAP["edge-add"],
                        EVENT_TYPE_ID_MAP["edge-delete"],
                        EVENT_TYPE_ID_MAP["node-add"],
                        EVENT_TYPE_ID_MAP["edge-delete"],
                        EVENT_TYPE_ID_MAP["end"],
                    ],
                ]
            ),
            torch.tensor([[0, 0, 1, 2, 2, 0, 0, 0], [0, 3, 4, 3, 4, 4, 3, 0]]),
            torch.tensor([list(range(8)), [i + 2 for i in range(8)]])
            .float()
            .unsqueeze(-1)
            .expand(-1, -1, 16),
            torch.tensor([[0] * 16, [0] * 16, [3] * 16, [5] * 16, [0] * 16]).float(),
        ),
    ],
)
def test_tgn_update_edge_features(
    event_type_ids, event_edge_ids, event_embeddings, expected
):
    tgn = TemporalGraphNetwork(5, 5, 10, 10, 10, 16, 10, 1, 1)
    tgn.update_edge_features(event_type_ids, event_edge_ids, event_embeddings)
    assert tgn.edge_features.equal(expected)


@pytest.mark.parametrize(
    "event_type_ids,event_edge_ids,event_timestamps,expected",
    [
        (
            torch.tensor(
                [
                    [
                        EVENT_TYPE_ID_MAP["start"],
                        EVENT_TYPE_ID_MAP["node-add"],
                        EVENT_TYPE_ID_MAP["node-delete"],
                        EVENT_TYPE_ID_MAP["end"],
                    ]
                ]
            ),
            torch.tensor([[0, 1, 2, 3]]).long(),
            torch.tensor([[0.0, 1.0, 2.0, 3.0]]).unsqueeze(-1),
            torch.zeros(5),
        ),
        (
            torch.tensor(
                [
                    [
                        EVENT_TYPE_ID_MAP["node-add"],
                        EVENT_TYPE_ID_MAP["node-delete"],
                        EVENT_TYPE_ID_MAP["edge-delete"],
                    ]
                ]
            ),
            torch.tensor([[1, 2, 3]]).long(),
            torch.tensor([[0.0, 1.0, 2.0]]).unsqueeze(-1),
            torch.tensor([0, 0, 0, 2, 0]).float(),
        ),
        (
            torch.tensor(
                [
                    [
                        EVENT_TYPE_ID_MAP["start"],
                        EVENT_TYPE_ID_MAP["edge-add"],
                        EVENT_TYPE_ID_MAP["end"],
                    ]
                ]
            ),
            torch.tensor([[0, 1, 0]]).long(),
            torch.tensor([[0.0, 1.0, 2.0]]).unsqueeze(-1),
            torch.tensor([0, 1, 0, 0, 0]).float(),
        ),
        (
            torch.tensor(
                [
                    [
                        EVENT_TYPE_ID_MAP["edge-add"],
                        EVENT_TYPE_ID_MAP["edge-delete"],
                    ]
                ]
            ),
            torch.tensor([[1, 1]]).long(),
            torch.tensor([[0.0, 1.0]]).unsqueeze(-1),
            torch.tensor([0, 1, 0, 0, 0]).float(),
        ),
        (
            torch.tensor(
                [
                    [
                        EVENT_TYPE_ID_MAP["start"],
                        EVENT_TYPE_ID_MAP["node-add"],
                        EVENT_TYPE_ID_MAP["node-delete"],
                        EVENT_TYPE_ID_MAP["edge-add"],
                        EVENT_TYPE_ID_MAP["edge-delete"],
                        EVENT_TYPE_ID_MAP["end"],
                        EVENT_TYPE_ID_MAP["pad"],
                        EVENT_TYPE_ID_MAP["pad"],
                    ],
                    [
                        EVENT_TYPE_ID_MAP["start"],
                        EVENT_TYPE_ID_MAP["node-add"],
                        EVENT_TYPE_ID_MAP["node-delete"],
                        EVENT_TYPE_ID_MAP["edge-add"],
                        EVENT_TYPE_ID_MAP["edge-delete"],
                        EVENT_TYPE_ID_MAP["node-add"],
                        EVENT_TYPE_ID_MAP["edge-delete"],
                        EVENT_TYPE_ID_MAP["end"],
                    ],
                ]
            ),
            torch.tensor([[0, 0, 1, 2, 2, 0, 0, 0], [0, 3, 4, 3, 4, 4, 3, 0]]),
            torch.tensor([list(range(8)), [i + 2 for i in range(8)]])
            .float()
            .unsqueeze(-1),
            torch.tensor([0, 0, 4, 8, 6]).float(),
        ),
    ],
)
def test_tgn_update_last_update(
    event_type_ids, event_edge_ids, event_timestamps, expected
):
    tgn = TemporalGraphNetwork(5, 5, 10, 10, 10, 16, 10, 1, 1)
    tgn.update_last_update(event_type_ids, event_edge_ids, event_timestamps)
    assert tgn.last_update.equal(expected)


@pytest.mark.parametrize(
    "event_type_dim,memory_dim,time_enc_dim,event_embedding_dim,output_dim,"
    "batch,event_seq_len,num_node,num_edge,transformer_conv_num_block,"
    "transformer_conv_num_heads",
    [
        (4, 8, 16, 20, 12, 1, 0, 0, 0, 1, 1),
        (4, 8, 16, 20, 12, 1, 3, 4, 5, 1, 1),
        (4, 8, 16, 20, 12, 1, 3, 4, 5, 2, 2),
        (4, 8, 16, 20, 12, 8, 5, 10, 20, 4, 4),
        (4, 8, 16, 20, 12, 8, 5, 10, 20, 4, 4),
        (8, 16, 32, 48, 24, 16, 10, 20, 40, 6, 6),
        (8, 16, 32, 48, 24, 16, 10, 20, 40, 8, 8),
    ],
)
def test_tgn_forward(
    event_type_dim,
    memory_dim,
    time_enc_dim,
    event_embedding_dim,
    output_dim,
    batch,
    event_seq_len,
    num_node,
    num_edge,
    transformer_conv_num_block,
    transformer_conv_num_heads,
):
    tgn = TemporalGraphNetwork(
        100,
        200,
        event_type_dim,
        memory_dim,
        time_enc_dim,
        event_embedding_dim,
        output_dim,
        transformer_conv_num_block,
        transformer_conv_num_heads,
    )
    assert (
        tgn(
            torch.randint(len(EVENT_TYPES), (batch, event_seq_len)),
            torch.randint(num_node, (batch, event_seq_len))
            if num_node > 0
            else torch.zeros((batch, event_seq_len)).long(),
            torch.randint(2, (batch, event_seq_len)).float(),
            torch.randint(num_node, (batch, event_seq_len))
            if num_node > 0
            else torch.zeros(batch, event_seq_len).long(),
            torch.randint(2, (batch, event_seq_len)).float(),
            torch.randint(num_edge, (batch, event_seq_len))
            if num_edge > 0
            else torch.zeros(batch, event_seq_len).long(),
            torch.rand(batch, event_seq_len, event_embedding_dim),
            torch.randint(10, (batch, event_seq_len)).float(),
            torch.randint(num_node, (batch, num_node))
            if num_node > 0
            else torch.zeros(batch, num_node).long(),
            torch.randint(num_edge, (batch, num_edge))
            if num_edge > 0
            else torch.zeros(batch, num_edge).long(),
            torch.randint(num_node, (batch, 2, num_edge))
            if num_node > 0
            else torch.zeros(batch, 2, num_edge).long(),
            torch.randint(10, (batch, num_edge)).float(),
        ).size()
        == (batch, num_node, output_dim)
    )


@pytest.mark.parametrize(
    "edge_index,node_ids,expected",
    [
        (
            torch.tensor([[[0], [0]]]),
            torch.tensor([[0]]),
            torch.tensor([[[0], [0]]]),
        ),
        (
            torch.tensor([[[0, 1], [0, 2]]]),
            torch.tensor([[0, 1, 2]]),
            torch.tensor([[[0, 1], [0, 2]]]),
        ),
        (
            torch.tensor([[[0, 1], [0, 2]]]),
            torch.tensor([[0, 2, 1]]),
            torch.tensor([[[0, 2], [0, 1]]]),
        ),
        (
            torch.tensor(
                [[[0, 5, 7, 5], [0, 9, 10, 4]], [[0, 8, 3, 8], [0, 6, 13, 12]]]
            ),
            torch.tensor([[0, 4, 5, 7, 9, 10], [0, 3, 6, 8, 12, 13]]),
            torch.tensor(
                [[[0, 2, 3, 2], [0, 4, 5, 1]], [[0, 9, 7, 9], [0, 8, 11, 10]]]
            ),
        ),
    ],
)
def test_localize_edge_index(edge_index, node_ids, expected):
    assert TemporalGraphNetwork.localize_edge_index(edge_index, node_ids).equal(
        expected
    )


@pytest.mark.parametrize(
    "node_dim,output_dim,num_block,heads,edge_dim,num_node,num_edge",
    [(16, 8, 1, 1, None, 1, 0), (16, 8, 4, 3, 12, 5, 4)],
)
def test_transformer_conv_stack(
    node_dim, output_dim, num_block, heads, edge_dim, num_node, num_edge
):
    stack = TransformerConvStack(
        node_dim, output_dim, num_block, heads=heads, edge_dim=edge_dim
    )
    assert (
        stack(
            torch.rand(num_node, node_dim),
            torch.randint(num_node, (2, num_edge)),
            edge_attr=None if edge_dim is None else torch.rand(num_edge, edge_dim),
        ).size()
        == (num_node, output_dim)
    )
