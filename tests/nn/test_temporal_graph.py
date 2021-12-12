import pytest
import torch

from torch_geometric.data import Batch

from dgu.nn.temporal_graph import TemporalGraphNetwork, TransformerConvStack


@pytest.mark.parametrize(
    "time_enc_dim,event_embedding_dim,output_dim,"
    "transformer_conv_num_block,transformer_conv_num_heads,batched_graph,num_node",
    [
        (
            16,
            20,
            12,
            1,
            1,
            Batch(
                batch=torch.tensor([0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3]),
                x=torch.rand(11, 20),
                node_last_update=torch.rand(11, 2),
                edge_index=torch.empty(2, 0).long(),
                edge_attr=torch.empty(0, 20),
                edge_last_update=torch.empty(0, 2),
            ),
            11,
        ),
        (
            16,
            20,
            12,
            1,
            1,
            Batch(
                batch=torch.tensor([0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3]),
                x=torch.rand(11, 20),
                node_last_update=torch.rand(11, 2),
                edge_index=torch.tensor([[5, 8, 10], [6, 9, 9]]),
                edge_attr=torch.rand(3, 20),
                edge_last_update=torch.rand(3, 2),
            ),
            11,
        ),
        (
            32,
            48,
            24,
            6,
            8,
            Batch(
                batch=torch.tensor([0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3]),
                x=torch.rand(11, 48),
                node_last_update=torch.rand(11, 2),
                edge_index=torch.tensor([[5, 8, 10], [6, 9, 9]]),
                edge_attr=torch.rand(3, 48),
                edge_last_update=torch.rand(3, 2),
            ),
            11,
        ),
    ],
)
@pytest.mark.parametrize("dropout", [None, 0.0, 0.3, 0.5])
def test_tgn_forward(
    dropout,
    time_enc_dim,
    event_embedding_dim,
    output_dim,
    transformer_conv_num_block,
    transformer_conv_num_heads,
    batched_graph,
    num_node,
):
    if dropout is None:
        tgn = TemporalGraphNetwork(
            time_enc_dim,
            event_embedding_dim,
            output_dim,
            transformer_conv_num_block,
            transformer_conv_num_heads,
        )
    else:
        tgn = TemporalGraphNetwork(
            time_enc_dim,
            event_embedding_dim,
            output_dim,
            transformer_conv_num_block,
            transformer_conv_num_heads,
            dropout=dropout,
        )
    node_embeddings = tgn(
        torch.randint(10, (batched_graph.num_graphs, 2)).float(), batched_graph
    )
    assert node_embeddings.size() == (num_node, output_dim)


@pytest.mark.parametrize(
    "node_dim,output_dim,num_block,heads,edge_dim,num_node,num_edge",
    [(16, 8, 1, 1, None, 1, 0), (16, 8, 4, 3, 12, 5, 4)],
)
@pytest.mark.parametrize("dropout", [None, 0.0, 0.3, 0.5])
def test_transformer_conv_stack_forward(
    dropout, node_dim, output_dim, num_block, heads, edge_dim, num_node, num_edge
):
    if dropout is None:
        stack = TransformerConvStack(
            node_dim, output_dim, num_block, heads=heads, edge_dim=edge_dim
        )
    else:
        stack = TransformerConvStack(
            node_dim,
            output_dim,
            num_block,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
        )
    assert (
        stack(
            torch.rand(num_node, node_dim),
            torch.randint(num_node, (2, num_edge)),
            edge_attr=None if edge_dim is None else torch.rand(num_edge, edge_dim),
        ).size()
        == (num_node, output_dim)
    )


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
