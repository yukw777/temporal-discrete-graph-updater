import pytest
import torch

from torch_geometric.data import Batch

from dgu.nn.dynamic_gnn import DynamicGNN, TransformerConvStack, ZeroPositionalEncoder


@pytest.mark.parametrize(
    "timestamp_enc_dim,event_embedding_dim,output_dim,transformer_conv_num_block,"
    "transformer_conv_num_heads,batched_graph,num_node",
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
                node_last_update=torch.randint(10, (11, 2)),
                edge_index=torch.empty(2, 0).long(),
                edge_attr=torch.empty(0, 20),
                edge_last_update=torch.empty(0, 2).long(),
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
                node_last_update=torch.randint(10, (11, 2)),
                edge_index=torch.tensor([[5, 8, 10], [6, 9, 9]]),
                edge_attr=torch.rand(3, 20),
                edge_last_update=torch.randint(10, (3, 2)),
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
                node_last_update=torch.randint(10, (11, 2)),
                edge_index=torch.tensor([[5, 8, 10], [6, 9, 9]]),
                edge_attr=torch.rand(3, 48),
                edge_last_update=torch.randint(10, (3, 2)),
            ),
            11,
        ),
    ],
)
@pytest.mark.parametrize("dropout", [None, 0.0, 0.3, 0.5])
@pytest.mark.parametrize("zero_timestamp_encoder", [True, False])
def test_dgnn_forward(
    zero_timestamp_encoder,
    dropout,
    timestamp_enc_dim,
    event_embedding_dim,
    output_dim,
    transformer_conv_num_block,
    transformer_conv_num_heads,
    batched_graph,
    num_node,
):
    if dropout is None:
        dgnn = DynamicGNN(
            timestamp_enc_dim,
            event_embedding_dim,
            output_dim,
            transformer_conv_num_block,
            transformer_conv_num_heads,
            zero_timestamp_encoder=zero_timestamp_encoder,
        )
    else:
        dgnn = DynamicGNN(
            timestamp_enc_dim,
            event_embedding_dim,
            output_dim,
            transformer_conv_num_block,
            transformer_conv_num_heads,
            dropout=dropout,
            zero_timestamp_encoder=zero_timestamp_encoder,
        )
    node_embeddings = dgnn(batched_graph)
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
    "channels,input_size", [(8, (2,)), (8, (2, 3)), (16, (2,)), (16, (2, 3))]
)
def test_zero_positional_encoder(channels, input_size):
    enc = ZeroPositionalEncoder(channels)
    encoded = enc(torch.rand(*input_size))
    assert encoded.size() == input_size + (channels,)
    assert torch.all(encoded == 0)
