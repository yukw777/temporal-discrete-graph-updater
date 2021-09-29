import pytest
import torch
import torch.nn as nn

from torch_geometric.data import Batch

from dgu.nn.temporal_graph import TemporalGraphNetwork, TransformerConvStack
from dgu.constants import EVENT_TYPES, EVENT_TYPE_ID_MAP


class MockTimeEncoder(nn.Module):
    def forward(self, timestamp):
        return timestamp.unsqueeze(-1).expand(-1, 4) + 3


@pytest.mark.parametrize(
    "event_type_ids,event_src_ids,event_dst_ids,event_embeddings,event_timestamps,"
    "node_id_offsets,memory,edge_index,edge_last_update,expected_node_add_event_msgs,"
    "expected_edge_event_node_ids,expected_edge_event_batch,"
    "expected_edge_event_messages",
    [
        (
            torch.tensor([EVENT_TYPE_ID_MAP["node-add"]]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([[3] * 4]).float(),
            torch.tensor([1.0]),
            torch.tensor([0]),
            torch.tensor([[1] * 4, [2] * 4]).float(),
            torch.empty(2, 0).long(),
            torch.empty(0),
            torch.tensor([[3] * 4 + [0] * 4 + [0] * 4 + [4] * 4 + [3] * 4]).float(),
            torch.empty(0).long(),
            torch.empty(0).long(),
            torch.empty(0, 20),
        ),
        (
            torch.tensor([EVENT_TYPE_ID_MAP["node-delete"]]),
            torch.tensor([1]),
            torch.tensor([0]),
            torch.tensor([[1] * 4]).float(),
            torch.tensor([4.0]),
            torch.tensor([0]),
            torch.tensor([[1] * 4, [2] * 4]).float(),
            torch.empty(2, 0).long(),
            torch.empty(0),
            torch.empty(0, 20),
            torch.empty(0).long(),
            torch.empty(0).long(),
            torch.empty(0, 20),
        ),
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["node-delete"],
                    EVENT_TYPE_ID_MAP["node-add"],
                ]
            ),
            torch.tensor([0, 2, 0]),
            torch.tensor([0, 0, 0]),
            torch.tensor([[1] * 4, [2] * 4, [3] * 4]).float(),
            torch.tensor([1.0, 2.0, 2.0]),
            torch.tensor([0, 2, 2]),
            torch.tensor([[4] * 4, [5] * 4]).float(),
            torch.tensor([[1], [0]]),
            torch.tensor([1.0]),
            torch.tensor(
                [
                    [3] * 4 + [0] * 4 + [0] * 4 + [4] * 4 + [1] * 4,
                    [3] * 4 + [0] * 4 + [0] * 4 + [5] * 4 + [3] * 4,
                ]
            ).float(),
            torch.empty(0).long(),
            torch.empty(0).long(),
            torch.empty(0, 20),
        ),
        (
            torch.tensor([EVENT_TYPE_ID_MAP["edge-add"]]),
            torch.tensor([1]),
            torch.tensor([0]),
            torch.tensor([[3] * 4]).float(),
            torch.tensor([2.0]),
            torch.tensor([0]),
            torch.tensor([[1] * 4, [2] * 4, [3] * 4]).float(),
            torch.tensor([[2], [1]]),
            torch.tensor([3.0]),
            torch.empty(0, 20),
            torch.tensor([1, 0]),
            torch.tensor([0, 0]),
            torch.tensor(
                [
                    [5] * 4 + [2] * 4 + [1] * 4 + [5] * 4 + [3] * 4,
                    [5] * 4 + [1] * 4 + [2] * 4 + [5] * 4 + [3] * 4,
                ]
            ).float(),
        ),
        (
            torch.tensor([EVENT_TYPE_ID_MAP["edge-delete"]]),
            torch.tensor([0]),
            torch.tensor([1]),
            torch.tensor([[2] * 4]).float(),
            torch.tensor([3.0]),
            torch.tensor([0]),
            torch.tensor([[1] * 4, [2] * 4, [3] * 4]).float(),
            torch.tensor([[2, 0], [1, 1]]),
            torch.tensor([2.0, 1.0]),
            torch.empty(0, 20),
            torch.tensor([0, 1]),
            torch.tensor([0, 0]),
            torch.tensor(
                [
                    [6] * 4 + [1] * 4 + [2] * 4 + [5] * 4 + [2] * 4,
                    [6] * 4 + [2] * 4 + [1] * 4 + [5] * 4 + [2] * 4,
                ]
            ).float(),
        ),
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                ]
            ),
            torch.tensor([0, 2, 1]),
            torch.tensor([1, 0, 2]),
            torch.tensor([[8] * 4, [9] * 4, [10] * 4]).float(),
            torch.tensor([2.0, 5.0, 3.0]),
            torch.tensor([0, 2, 5]),
            torch.tensor(
                [[1] * 4, [2] * 4, [3] * 4, [4] * 4, [5] * 4, [6] * 4, [7] * 4, [8] * 4]
            ).float(),
            torch.tensor([[1, 4, 5], [0, 2, 6]]),
            torch.tensor([3.0, 2.0, 4.0]),
            torch.empty(0, 20),
            torch.tensor([0, 2, 1, 1, 0, 2]),
            torch.tensor([0, 1, 2, 0, 1, 2]),
            torch.tensor(
                [
                    [5] * 4 + [1] * 4 + [2] * 4 + [5] * 4 + [8] * 4,
                    [6] * 4 + [5] * 4 + [3] * 4 + [6] * 4 + [9] * 4,
                    [5] * 4 + [7] * 4 + [8] * 4 + [6] * 4 + [10] * 4,
                    [5] * 4 + [2] * 4 + [1] * 4 + [5] * 4 + [8] * 4,
                    [6] * 4 + [3] * 4 + [5] * 4 + [6] * 4 + [9] * 4,
                    [5] * 4 + [8] * 4 + [7] * 4 + [6] * 4 + [10] * 4,
                ]
            ).float(),
        ),
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                    EVENT_TYPE_ID_MAP["node-delete"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["node-add"],
                ]
            ),
            torch.tensor([0, 0, 2, 1, 1, 0]),
            torch.tensor([1, 0, 0, 0, 2, 0]),
            torch.tensor(
                [[2] * 4, [3] * 4, [4] * 4, [5] * 4, [6] * 4, [7] * 4]
            ).float(),
            torch.tensor([2.0, 5.0, 3.0, 2.0, 1.0, 4.0]),
            torch.tensor([0, 2, 2, 5, 7, 10]),
            # torch.tensor([0, 0, 2, 2, 2, 3, 3, 4, 4, 4]),
            torch.tensor(
                [
                    [1] * 4,
                    [2] * 4,
                    [3] * 4,
                    [4] * 4,
                    [5] * 4,
                    [6] * 4,
                    [7] * 4,
                    [8] * 4,
                    [9] * 4,
                    [10] * 4,
                ]
            ).float(),
            torch.tensor([[1, 4], [0, 2]]),
            torch.tensor([3.0, 2.0]),
            torch.tensor(
                [
                    [3] * 4 + [0] * 4 + [0] * 4 + [8] * 4 + [3] * 4,
                    [3] * 4 + [0] * 4 + [0] * 4 + [7] * 4 + [7] * 4,
                ]
            ).float(),
            torch.tensor([0, 2, 1, 1, 0, 2]),
            torch.tensor([0, 2, 4, 0, 2, 4]),
            torch.tensor(
                [
                    [5] * 4 + [1] * 4 + [2] * 4 + [5] * 4 + [2] * 4,
                    [6] * 4 + [5] * 4 + [3] * 4 + [4] * 4 + [4] * 4,
                    [5] * 4 + [9] * 4 + [10] * 4 + [4] * 4 + [6] * 4,
                    [5] * 4 + [2] * 4 + [1] * 4 + [5] * 4 + [2] * 4,
                    [6] * 4 + [3] * 4 + [5] * 4 + [4] * 4 + [4] * 4,
                    [5] * 4 + [10] * 4 + [9] * 4 + [4] * 4 + [6] * 4,
                ]
            ).float(),
        ),
    ],
)
def test_tgn_get_event_message(
    event_type_ids,
    event_src_ids,
    event_dst_ids,
    event_embeddings,
    event_timestamps,
    node_id_offsets,
    memory,
    edge_index,
    edge_last_update,
    expected_node_add_event_msgs,
    expected_edge_event_node_ids,
    expected_edge_event_batch,
    expected_edge_event_messages,
):
    tgn = TemporalGraphNetwork(4, 4, 4, 4, 8, 1, 1)
    tgn.time_encoder = MockTimeEncoder()
    tgn.event_type_emb = nn.Embedding.from_pretrained(
        torch.linspace(0, 6, 7).unsqueeze(-1).expand(-1, 4)
    )

    (
        node_add_event_msgs,
        edge_event_node_ids,
        edge_event_batch,
        edge_event_messages,
    ) = tgn.get_event_messages(
        event_type_ids,
        event_src_ids,
        event_dst_ids,
        event_embeddings,
        event_timestamps,
        node_id_offsets,
        memory,
        edge_index,
        edge_last_update,
    )
    assert node_add_event_msgs.equal(expected_node_add_event_msgs)
    assert edge_event_node_ids.equal(expected_edge_event_node_ids)
    assert edge_event_batch.equal(expected_edge_event_batch)
    assert edge_event_messages.equal(expected_edge_event_messages)


@pytest.mark.parametrize(
    "event_type_dim,memory_dim,time_enc_dim,event_embedding_dim,output_dim,"
    "num_edge_event,prev_num_node,num_node,num_edge,"
    "transformer_conv_num_block,transformer_conv_num_heads",
    [
        (4, 8, 16, 20, 12, 0, 0, 0, 0, 1, 1),
        (4, 8, 16, 20, 12, 0, 1, 1, 0, 1, 1),
        (4, 8, 16, 20, 12, 1, 2, 2, 1, 1, 1),
        (4, 8, 16, 20, 12, 4, 8, 5, 10, 4, 4),
        (8, 16, 32, 48, 24, 10, 10, 8, 10, 6, 6),
        (8, 16, 32, 48, 24, 10, 6, 8, 10, 6, 6),
        (8, 16, 32, 48, 24, 10, 12, 8, 10, 8, 8),
    ],
)
def test_tgn_forward(
    event_type_dim,
    memory_dim,
    time_enc_dim,
    event_embedding_dim,
    output_dim,
    num_edge_event,
    prev_num_node,
    num_node,
    num_edge,
    transformer_conv_num_block,
    transformer_conv_num_heads,
):
    tgn = TemporalGraphNetwork(
        event_type_dim,
        memory_dim,
        time_enc_dim,
        event_embedding_dim,
        output_dim,
        transformer_conv_num_block,
        transformer_conv_num_heads,
    )
    edge_index = (
        torch.randint(num_node, (2, num_edge)).unique(dim=1)
        if num_node > 0 and num_edge > 1
        else torch.zeros(2, num_edge).long()
    )
    results = tgn(
        torch.randint(len(EVENT_TYPES), (num_edge_event,)),
        torch.randint(prev_num_node, (num_edge_event,))
        if prev_num_node > 0
        else torch.zeros(num_edge_event).long(),
        torch.randint(prev_num_node, (num_edge_event,))
        if prev_num_node > 0
        else torch.zeros(num_edge_event).long(),
        torch.rand(num_edge_event, event_embedding_dim),
        torch.randint(10, (num_edge_event,)).float(),
        torch.rand(prev_num_node, memory_dim),
        torch.randint(num_node, (prev_num_node,))
        if num_node > 0
        else torch.zeros(prev_num_node).long(),
        torch.randint(2, (prev_num_node,), dtype=torch.bool)
        if num_node > 0
        else torch.zeros(prev_num_node).bool(),
        torch.rand(num_node, event_embedding_dim),
        edge_index,
        torch.rand(edge_index.size(1), event_embedding_dim),
        torch.randint(10, (edge_index.size(1),)).float(),
        torch.randint(10, (edge_index.size(1),)).float(),
    )
    assert results["node_embeddings"].size() == (num_node, output_dim)
    assert results["updated_memory"].size() == (num_node, memory_dim)


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


class MockRNN(nn.Module):
    def forward(self, input, hidden):
        return input[:, 4:20:4] + hidden


@pytest.mark.parametrize(
    "batch_size,batch,expected",
    [
        (1, torch.empty(0).long(), torch.tensor([0])),
        (1, torch.tensor([0, 0, 0]), torch.tensor([0])),
        (3, torch.tensor([0, 1, 1, 2, 2, 2]), torch.tensor([0, 1, 3])),
        (5, torch.tensor([0, 2, 2, 3, 3, 3]), torch.tensor([0, 1, 1, 3, 6])),
    ],
)
def test_tgn_calculate_node_id_offsets(batch_size, batch, expected):
    node_id_offsets = TemporalGraphNetwork.calculate_node_id_offsets(batch_size, batch)
    assert node_id_offsets.equal(expected)


@pytest.mark.parametrize(
    "batched_graphs,memory,event_type_ids,event_src_ids,event_dst_ids,event_embeddings,"
    "event_timestamps,expected_updated_batched_graph,expected_updated_memory",
    [
        (
            Batch(
                batch=torch.empty(0).long(),
                x=torch.empty(0, 4),
                edge_index=torch.empty(2, 0).long(),
                edge_attr=torch.empty(0, 4),
                edge_last_update=torch.empty(0),
            ),
            torch.empty(0, 4),
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["pad"],
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["end"],
                ]
            ),
            torch.zeros(3).long(),
            torch.zeros(3).long(),
            torch.zeros(3, 4),
            torch.zeros(3),
            Batch(
                batch=torch.empty(0).long(),
                x=torch.empty(0, 4),
                edge_index=torch.empty(2, 0).long(),
                edge_attr=torch.empty(0, 4),
                edge_last_update=torch.empty(0),
            ),
            torch.empty(0, 4),
        ),
        (
            Batch(
                batch=torch.tensor([0, 0, 1, 1]),
                x=torch.tensor([[4] * 4, [3] * 4, [2] * 4, [1] * 4]).float(),
                edge_index=torch.empty(2, 0).long(),
                edge_attr=torch.empty(0, 4),
                edge_last_update=torch.empty(0),
            ),
            torch.tensor([[5] * 4, [6] * 4, [7] * 4, [8] * 4]).float(),
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                ]
            ),
            torch.tensor([0, 1]),
            torch.tensor([0, 0]),
            torch.tensor([[5] * 4, [6] * 4]).float(),
            torch.tensor([3.0, 5.0]),
            Batch(
                batch=torch.tensor([0, 0, 0, 1, 1]),
                x=torch.tensor([[4] * 4, [3] * 4, [5] * 4, [2] * 4, [1] * 4]).float(),
                edge_index=torch.tensor([[4], [3]]),
                edge_attr=torch.tensor([[6] * 4]).float(),
                edge_last_update=torch.tensor([5.0]),
            ),
            torch.tensor(
                [[5] * 4, [6] * 4, [0, 0, 6, 5], [14, 15, 15, 13], [16, 15, 16, 14]]
            ).float(),
        ),
        (
            Batch(
                batch=torch.tensor([0, 0, 0, 1, 1]),
                x=torch.tensor([[4] * 4, [3] * 4, [5] * 4, [2] * 4, [1] * 4]).float(),
                edge_index=torch.tensor([[4], [3]]),
                edge_attr=torch.tensor([[6] * 4]).float(),
                edge_last_update=torch.tensor([5.0]),
            ),
            torch.tensor([[5] * 4, [6] * 4, [7] * 4, [8] * 4, [9] * 4]).float(),
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["node-delete"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                ]
            ),
            torch.tensor([2, 1]),
            torch.tensor([0, 0]),
            torch.tensor([[5] * 4, [6] * 4]),
            torch.tensor([2.0, 3.0]),
            Batch(
                batch=torch.tensor([0, 0, 1, 1]),
                x=torch.tensor([[4] * 4, [3] * 4, [2] * 4, [1] * 4]).float(),
                edge_index=torch.empty(2, 0).long(),
                edge_attr=torch.empty(0, 4),
                edge_last_update=torch.empty(0),
            ),
            torch.tensor([[5] * 4, [6] * 4, [16, 17, 9, 14], [18, 17, 10, 15]]).float(),
        ),
        (
            Batch(
                batch=torch.tensor([0, 0, 1, 1]),
                x=torch.tensor([[4] * 4, [3] * 4, [2] * 4, [1] * 4]).float(),
                edge_index=torch.empty(2, 0).long(),
                edge_attr=torch.empty(0, 4),
                edge_last_update=torch.empty(0),
            ),
            torch.tensor([[5] * 4, [6] * 4, [7] * 4, [8] * 4]).float(),
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["node-add"],
                ]
            ),
            torch.tensor([0, 1, 0]),
            torch.tensor([0, 0, 0]),
            torch.tensor([[5] * 4, [6] * 4, [7] * 4]).float(),
            torch.tensor([3.0, 5.0, 4.0]),
            Batch(
                batch=torch.tensor([0, 0, 0, 1, 1, 2]),
                x=torch.tensor(
                    [[4] * 4, [3] * 4, [5] * 4, [2] * 4, [1] * 4, [7] * 4]
                ).float(),
                edge_index=torch.tensor([[4], [3]]),
                edge_attr=torch.tensor([[6] * 4]).float(),
                edge_last_update=torch.tensor([5.0]),
            ),
            torch.tensor(
                [
                    [5] * 4,
                    [6] * 4,
                    [0, 0, 6, 5],
                    [14, 15, 15, 13],
                    [16, 15, 16, 14],
                    [0, 0, 7, 7],
                ]
            ).float(),
        ),
        (
            Batch(
                batch=torch.tensor([0, 0, 0, 1, 1, 2]),
                x=torch.tensor(
                    [[4] * 4, [3] * 4, [5] * 4, [2] * 4, [1] * 4, [7] * 4]
                ).float(),
                edge_index=torch.tensor([[4], [3]]),
                edge_attr=torch.tensor([[6] * 4]).float(),
                edge_last_update=torch.tensor([5.0]),
            ),
            torch.tensor(
                [[3] * 4, [4] * 4, [5] * 4, [6] * 4, [7] * 4, [8] * 4]
            ).float(),
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["node-delete"],
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["node-add"],
                ]
            ),
            torch.tensor([1, 0, 0]),
            torch.tensor([0, 0, 0]),
            torch.tensor([[3] * 4, [8] * 4, [9] * 4]).float(),
            torch.tensor([2.0, 3.0, 4.0]),
            Batch(
                batch=torch.tensor([0, 0, 1, 1, 1, 2, 2]),
                x=torch.tensor(
                    [[4] * 4, [5] * 4, [2] * 4, [1] * 4, [8] * 4, [7] * 4, [9] * 4]
                ).float(),
                edge_index=torch.tensor([[3], [2]]),
                edge_attr=torch.tensor([[6] * 4]).float(),
                edge_last_update=torch.tensor([5.0]),
            ),
            torch.tensor(
                [
                    [3] * 4,
                    [5] * 4,
                    [6] * 4,
                    [7] * 4,
                    [0, 0, 6, 8],
                    [8] * 4,
                    [0, 0, 7, 9],
                ]
            ).float(),
        ),
        (
            Batch(
                batch=torch.tensor([0, 0, 1, 1, 1, 2, 2]),
                x=torch.tensor(
                    [[4] * 4, [5] * 4, [2] * 4, [1] * 4, [8] * 4, [7] * 4, [9] * 4]
                ).float(),
                edge_index=torch.tensor([[3], [2]]),
                edge_attr=torch.tensor([[6] * 4]).float(),
                edge_last_update=torch.tensor([5.0]),
            ),
            torch.tensor(
                [[3] * 4, [4] * 4, [5] * 4, [6] * 4, [7] * 4, [8] * 4, [9] * 4]
            ).float(),
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                ]
            ),
            torch.tensor([0, 1, 0]),
            torch.tensor([0, 0, 1]),
            torch.tensor([[10] * 4, [6] * 4, [11] * 4]).float(),
            torch.tensor([5.0, 3.0, 2.0]),
            Batch(
                batch=torch.tensor([0, 0, 0, 1, 1, 1, 2, 2]),
                x=torch.tensor(
                    [
                        [4] * 4,
                        [5] * 4,
                        [10] * 4,
                        [2] * 4,
                        [1] * 4,
                        [8] * 4,
                        [7] * 4,
                        [9] * 4,
                    ]
                ).float(),
                edge_index=torch.tensor([[6], [7]]),
                edge_attr=torch.tensor([[11] * 4]).float(),
                edge_last_update=torch.tensor([2.0]),
            ),
            torch.tensor(
                [
                    [3] * 4,
                    [4] * 4,
                    [0, 0, 8, 10],
                    [10, 11, 6, 11],
                    [12, 11, 7, 12],
                    [7] * 4,
                    [16, 17, 13, 19],
                    [18, 17, 14, 20],
                ]
            ).float(),
        ),
        (
            Batch(
                batch=torch.tensor([0, 0, 0, 1, 1, 1, 2, 2]),
                x=torch.tensor(
                    [
                        [4] * 4,
                        [5] * 4,
                        [10] * 4,
                        [2] * 4,
                        [1] * 4,
                        [8] * 4,
                        [7] * 4,
                        [9] * 4,
                    ]
                ).float(),
                edge_index=torch.tensor([[6], [7]]),
                edge_attr=torch.tensor([[11] * 4]).float(),
                edge_last_update=torch.tensor([2.0]),
            ),
            torch.tensor(
                [[2] * 4, [3] * 4, [4] * 4, [5] * 4, [6] * 4, [7] * 4, [8] * 4, [9] * 4]
            ).float(),
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["node-delete"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                ]
            ),
            torch.tensor([2, 1, 0]),
            torch.tensor([0, 0, 1]),
            torch.tensor([[12] * 4, [1] * 4, [11] * 4]).float(),
            torch.tensor([3.0, 1.0, 6.0]),
            Batch(
                batch=torch.tensor([0, 0, 0, 1, 1, 2, 2]),
                x=torch.tensor(
                    [[4] * 4, [5] * 4, [10] * 4, [2] * 4, [8] * 4, [7] * 4, [9] * 4]
                ).float(),
                edge_index=torch.tensor([[2], [0]]),
                edge_attr=torch.tensor([[12] * 4]).float(),
                edge_last_update=torch.tensor([3.0]),
            ),
            torch.tensor(
                [
                    [4, 6, 8, 14],
                    [3] * 4,
                    [8, 6, 10, 16],
                    [5] * 4,
                    [7] * 4,
                    [16, 17, 15, 19],
                    [18, 17, 16, 20],
                ]
            ).float(),
        ),
        (
            Batch(
                batch=torch.tensor([0, 0, 0, 1, 1, 2, 2]),
                x=torch.tensor(
                    [[4] * 4, [5] * 4, [10] * 4, [2] * 4, [8] * 4, [7] * 4, [9] * 4]
                ).float(),
                edge_index=torch.tensor([[2], [0]]),
                edge_attr=torch.tensor([[12] * 4]).float(),
                edge_last_update=torch.tensor([3.0]),
            ),
            torch.tensor(
                [[3] * 4, [4] * 4, [5] * 4, [6] * 4, [7] * 4, [8] * 4, [9] * 4]
            ).float(),
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["edge-delete"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                ]
            ),
            torch.tensor([2, 0, 1]),
            torch.tensor([0, 1, 0]),
            torch.tensor([[12] * 4, [13] * 4, [14] * 4]).float(),
            torch.tensor([2.0, 8.0, 7.0]),
            Batch(
                batch=torch.tensor([0, 0, 0, 1, 1, 2, 2]),
                x=torch.tensor(
                    [[4] * 4, [5] * 4, [10] * 4, [2] * 4, [8] * 4, [7] * 4, [9] * 4]
                ).float(),
                edge_index=torch.tensor([[3, 6], [4, 5]]),
                edge_attr=torch.tensor([[13] * 4, [14] * 4]).float(),
                edge_last_update=torch.tensor([8.0, 7.0]),
            ),
            torch.tensor(
                [
                    [6, 8, 5, 15],
                    [4] * 4,
                    [10, 8, 7, 17],
                    [12, 13, 17, 19],
                    [14, 13, 18, 20],
                    [16, 17, 18, 22],
                    [18, 17, 19, 23],
                ]
            ).float(),
        ),
        (
            Batch(
                batch=torch.tensor([0, 0, 0, 1, 1, 1]),
                x=torch.tensor(
                    [[3] * 4, [4] * 4, [5] * 4, [6] * 4, [7] * 4, [8] * 4]
                ).float(),
                edge_index=torch.tensor([[2, 5], [0, 3]]),
                edge_attr=torch.tensor([[1] * 4, [2] * 4]).float(),
                edge_last_update=torch.tensor([3.0, 2.0]),
            ),
            torch.tensor(
                [[1] * 4, [2] * 4, [3] * 4, [4] * 4, [5] * 4, [6] * 4]
            ).float(),
            torch.tensor(
                [EVENT_TYPE_ID_MAP["edge-add"], EVENT_TYPE_ID_MAP["edge-delete"]]
            ),
            torch.tensor([0, 2]),
            torch.tensor([1, 0]),
            torch.tensor([[7] * 4, [8] * 4]).float(),
            torch.tensor([5.0, 3.0]),
            Batch(
                batch=torch.tensor([0, 0, 0, 1, 1, 1]),
                x=torch.tensor(
                    [[3] * 4, [4] * 4, [5] * 4, [6] * 4, [7] * 4, [8] * 4]
                ).float(),
                edge_index=torch.tensor([[2, 0], [0, 1]]),
                edge_attr=torch.tensor([[1] * 4, [7] * 4]).float(),
                edge_last_update=torch.tensor([3.0, 5.0]),
            ),
            torch.tensor(
                [
                    [2, 3, 9, 8],
                    [4, 3, 10, 9],
                    [3] * 4,
                    [8, 10, 8, 12],
                    [5] * 4,
                    [12, 10, 10, 14],
                ]
            ).float(),
        ),
    ],
)
def test_tgn_update_batched_graph_memory(
    batched_graphs,
    memory,
    event_type_ids,
    event_src_ids,
    event_dst_ids,
    event_embeddings,
    event_timestamps,
    expected_updated_batched_graph,
    expected_updated_memory,
):
    tgn = TemporalGraphNetwork(4, 4, 4, 4, 8, 1, 1)
    tgn.time_encoder = MockTimeEncoder()
    tgn.event_type_emb = nn.Embedding.from_pretrained(
        torch.linspace(0, 6, 7).unsqueeze(-1).expand(-1, 4)
    )
    tgn.rnn = MockRNN()
    updated_batched_graph, updated_memory = tgn.update_batched_graph_memory(
        batched_graphs,
        memory,
        event_type_ids,
        event_src_ids,
        event_dst_ids,
        event_embeddings,
        event_timestamps,
    )
    assert updated_batched_graph.batch.equal(expected_updated_batched_graph.batch)
    assert updated_batched_graph.x.equal(expected_updated_batched_graph.x)
    assert updated_batched_graph.edge_index.equal(
        expected_updated_batched_graph.edge_index
    )
    assert updated_batched_graph.edge_attr.equal(
        expected_updated_batched_graph.edge_attr
    )
    assert updated_batched_graph.edge_last_update.equal(
        expected_updated_batched_graph.edge_last_update
    )
    assert updated_memory.equal(expected_updated_memory)
    assert updated_batched_graph.x.size(0) == updated_memory.size(0)


@pytest.mark.parametrize(
    "timestamps,batch,edge_index,expected",
    [
        (
            torch.tensor([2.0]),
            torch.empty(0).long(),
            torch.empty(2, 0).long(),
            torch.empty(0),
        ),
        (
            torch.tensor([2.0, 3.0, 4.0]),
            torch.empty(0).long(),
            torch.empty(2, 0).long(),
            torch.empty(0),
        ),
        (
            torch.tensor([2.0]),
            torch.tensor([0, 0, 0]),
            torch.tensor([[0, 2], [1, 1]]),
            torch.tensor([2.0, 2.0]),
        ),
        (
            torch.tensor([2.0, 3.0, 4.0]),
            torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2]),
            torch.tensor([[0, 2, 4, 6, 8], [1, 1, 3, 7, 5]]),
            torch.tensor([2.0, 2.0, 3.0, 4.0, 4.0]),
        ),
    ],
)
def test_sldgu_get_edge_timestamps(timestamps, batch, edge_index, expected):
    assert TemporalGraphNetwork.get_edge_timestamps(
        timestamps, batch, edge_index
    ).equal(expected)
