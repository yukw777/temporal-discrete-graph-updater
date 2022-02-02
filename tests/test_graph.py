import pytest
import torch
import networkx as nx

from torch_geometric.data import Data, Batch
from copy import deepcopy

from tdgu.graph import (
    Node,
    DstNode,
    ExitNode,
    FoodNameNode,
    FoodAdjNode,
    process_triplet_cmd,
    data_to_networkx,
    batch_to_data_list,
    networkx_to_rdf,
    update_rdf_graph,
)

from utils import EqualityDiGraph


@pytest.mark.parametrize(
    "graph,timestamp,cmd,expected",
    [
        # regular source and destination nodes
        (
            EqualityDiGraph(),
            0,
            "add , player , kitchen , in",
            [
                {"type": "node-add", "label": "player"},
                {"type": "node-add", "label": "kitchen"},
                {"type": "edge-add", "src_id": 0, "dst_id": 1, "label": "in"},
            ],
        ),
        # regular source and destination nodes already exist
        (
            EqualityDiGraph(
                {Node("player"): {Node("kitchen"): {"label": "in", "last_update": 0}}}
            ),
            0,
            "add , player , kitchen , in",
            [],
        ),
        # exit source node
        (
            EqualityDiGraph(),
            0,
            "add , exit , kitchen , west_of",
            [
                {"type": "node-add", "label": "exit"},
                {"type": "node-add", "label": "kitchen"},
                {"type": "edge-add", "src_id": 0, "dst_id": 1, "label": "west of"},
            ],
        ),
        # exit source node with a different relation exists
        (
            EqualityDiGraph(
                {
                    ExitNode("exit", "east of", "kitchen"): {
                        Node("kitchen"): {"label": "east of", "last_update": 0}
                    }
                }
            ),
            1,
            "add , exit , kitchen , west_of",
            [
                {"type": "node-add", "label": "exit"},
                {"type": "edge-add", "src_id": 2, "dst_id": 1, "label": "west of"},
            ],
        ),
        # is destination node
        (
            EqualityDiGraph(),
            0,
            "add , steak , cooked , is",
            [
                {"type": "node-add", "label": "steak"},
                {"type": "node-add", "label": "cooked"},
                {"type": "edge-add", "src_id": 0, "dst_id": 1, "label": "is"},
            ],
        ),
        # is destination node already exists
        (
            EqualityDiGraph(
                {
                    Node("steak"): {
                        DstNode("cooked", "steak"): {"label": "is", "last_update": 1}
                    }
                }
            ),
            2,
            "add , steak , delicious , is",
            [
                {"type": "node-add", "label": "delicious"},
                {"type": "edge-add", "src_id": 0, "dst_id": 2, "label": "is"},
            ],
        ),
        # regular source and destination nodes deletion
        (
            EqualityDiGraph(
                {Node("player"): {Node("kitchen"): {"label": "in", "last_update": 0}}}
            ),
            1,
            "delete , player , kitchen , in",
            [
                {
                    "type": "edge-delete",
                    "src_id": 0,
                    "dst_id": 1,
                    "label": "in",
                    "edge_id": 0,
                },
                {"type": "node-delete", "node_id": 1, "label": "kitchen"},
                {"type": "node-delete", "node_id": 0, "label": "player"},
            ],
        ),
        # source node doesn't exist
        (
            EqualityDiGraph(
                {Node("player"): {Node("kitchen"): {"label": "in", "last_update": 0}}}
            ),
            1,
            "delete , bogus , kitchen , in",
            [],
        ),
        # destination node doesn't exist
        (
            EqualityDiGraph(
                {Node("player"): {Node("kitchen"): {"label": "in", "last_update": 0}}}
            ),
            1,
            "delete , player , bogus , in",
            [],
        ),
        # relation doesn't exist
        (
            EqualityDiGraph({Node("player"): {}, Node("kitchen"): {}}),
            0,
            "delete , player , kitchen , bogus",
            [],
        ),
        # exit source node deletion
        (
            EqualityDiGraph(
                {
                    ExitNode("exit", "east of", "kitchen"): {
                        Node("kitchen"): {"label": "east of", "last_update": 2}
                    },
                    ExitNode("exit", "west of", "kitchen"): {
                        Node("kitchen"): {"label": "west of", "last_update": 2}
                    },
                }
            ),
            3,
            "delete , exit , kitchen , west_of",
            [
                {
                    "type": "edge-delete",
                    "src_id": 1,
                    "dst_id": 2,
                    "label": "west of",
                    "edge_id": 1,
                },
                {"type": "node-delete", "node_id": 1, "label": "exit"},
            ],
        ),
        # is destination node deletion
        (
            EqualityDiGraph(
                {
                    Node("steak"): {
                        DstNode("cooked", "steak"): {"label": "is", "last_update": 3},
                        DstNode("delicious", "steak"): {
                            "label": "is",
                            "last_update": 3,
                        },
                    }
                }
            ),
            4,
            "delete , steak , delicious , is",
            [
                {
                    "type": "edge-delete",
                    "src_id": 0,
                    "dst_id": 2,
                    "label": "is",
                    "edge_id": 1,
                },
                {
                    "type": "node-delete",
                    "node_id": 2,
                    "label": "delicious",
                },
            ],
        ),
    ],
)
@pytest.mark.parametrize("allow_objs_with_same_label", [False, True])
def test_process_triplet_cmd(
    allow_objs_with_same_label, graph, timestamp, cmd, expected
):
    assert (
        process_triplet_cmd(
            # need to deep copy the graph to avoid side effects from previous tests
            deepcopy(graph),
            timestamp,
            cmd,
            allow_objs_with_same_label=allow_objs_with_same_label,
        )
        == expected
    )


@pytest.mark.parametrize(
    "graph,timestamp,cmd,expected",
    [
        (
            EqualityDiGraph(),
            0,
            "add , red onion , counter , on",
            [
                {"type": "node-add", "label": "onion"},
                {"type": "node-add", "label": "red"},
                {"type": "edge-add", "src_id": 0, "dst_id": 1, "label": "is"},
                {"type": "node-add", "label": "counter"},
                {"type": "edge-add", "src_id": 0, "dst_id": 2, "label": "on"},
            ],
        ),
        (
            EqualityDiGraph(
                {
                    FoodNameNode("onion", "red onion"): {
                        FoodAdjNode("red", FoodNameNode("onion", "red onion")): {
                            "label": "is",
                            "last_update": 0,
                        },
                        Node("counter"): {"label": "on", "last_update": 0},
                    }
                }
            ),
            1,
            "add , red onion , player , in",
            [
                {"type": "node-add", "label": "player"},
                {"type": "edge-add", "src_id": 0, "dst_id": 3, "label": "in"},
            ],
        ),
        (
            EqualityDiGraph(
                {
                    FoodNameNode("onion", "red onion"): {
                        FoodAdjNode("red", FoodNameNode("onion", "red onion")): {
                            "label": "is",
                            "last_update": 0,
                        },
                        Node("counter"): {"label": "on", "last_update": 0},
                    }
                }
            ),
            1,
            "delete , red onion , counter , on",
            [
                {
                    "type": "edge-delete",
                    "src_id": 0,
                    "dst_id": 2,
                    "label": "on",
                    "edge_id": 1,
                },
                {"type": "node-delete", "node_id": 2, "label": "counter"},
                {
                    "type": "edge-delete",
                    "src_id": 0,
                    "dst_id": 1,
                    "label": "is",
                    "edge_id": 0,
                },
                {"type": "node-delete", "node_id": 1, "label": "red"},
                {"type": "node-delete", "node_id": 0, "label": "onion"},
            ],
        ),
    ],
)
def test_process_triplet_cmd_allow_objs_with_same_label(
    graph, timestamp, cmd, expected
):
    processed = process_triplet_cmd(
        graph, timestamp, cmd, allow_objs_with_same_label=True
    )
    assert processed == expected


@pytest.mark.parametrize(
    "data,labels,expected_node_attrs,expected_edge_attrs",
    [
        (
            Data(
                x=torch.empty(0, dtype=torch.long),
                node_last_update=torch.empty(0, 2),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=torch.empty(0, dtype=torch.long),
                edge_last_update=torch.empty(0, 2),
            ),
            ["", "player", "livingroom", "table", "at", "in"],
            {},
            [],
        ),
        (
            Data(
                x=torch.tensor([1]),
                node_last_update=torch.tensor([[1, 0]]),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=torch.empty(0, dtype=torch.long),
                edge_last_update=torch.empty(0, 2),
            ),
            ["", "player", "livingroom", "table", "at", "in"],
            {0: {"x": 1, "node_last_update": [1, 0], "label": "player"}},
            [],
        ),
        (
            Data(
                x=torch.tensor([1, 2]),
                node_last_update=torch.tensor([[1, 0], [1, 1]]),
                edge_index=torch.tensor([[0], [1]]),
                edge_attr=torch.tensor([4]),
                edge_last_update=torch.tensor([[1, 2]]),
            ),
            ["", "player", "livingroom", "table", "at", "in"],
            {
                0: {"x": 1, "node_last_update": [1, 0], "label": "player"},
                1: {"x": 2, "node_last_update": [1, 1], "label": "livingroom"},
            },
            [
                (0, 1, {"edge_attr": 4, "edge_last_update": [1, 2], "label": "at"}),
            ],
        ),
        (
            Data(
                x=torch.tensor([1, 2]),
                node_last_update=torch.tensor([[1, 0], [1, 1]]),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=torch.empty(0, dtype=torch.long),
                edge_last_update=torch.empty(0, 2),
            ),
            ["", "player", "livingroom", "table", "at", "in"],
            {
                0: {"x": 1, "node_last_update": [1, 0], "label": "player"},
                1: {"x": 2, "node_last_update": [1, 1], "label": "livingroom"},
            },
            [],
        ),
        (
            Data(
                x=torch.tensor([1, 2, 3]),
                node_last_update=torch.tensor([[1, 0], [1, 1], [2, 3]]),
                edge_index=torch.tensor([[0, 2], [1, 1]]),
                edge_attr=torch.tensor([4, 5]),
                edge_last_update=torch.tensor([[1, 2], [2, 2]]),
            ),
            ["", "player", "livingroom", "table", "at", "in"],
            {
                0: {"x": 1, "node_last_update": [1, 0], "label": "player"},
                1: {"x": 2, "node_last_update": [1, 1], "label": "livingroom"},
                2: {"x": 3, "node_last_update": [2, 3], "label": "table"},
            },
            [
                (0, 1, {"edge_attr": 4, "edge_last_update": [1, 2], "label": "at"}),
                (2, 1, {"edge_attr": 5, "edge_last_update": [2, 2], "label": "in"}),
            ],
        ),
    ],
)
def test_data_to_networkx(data, labels, expected_node_attrs, expected_edge_attrs):
    nx_graph = data_to_networkx(data, labels)
    assert dict(nx_graph.nodes.data()) == expected_node_attrs
    assert list(nx_graph.edges.data()) == expected_edge_attrs


@pytest.mark.parametrize(
    "batch,batch_size,expected_list",
    [
        (
            Batch(
                batch=torch.empty(0, dtype=torch.long),
                x=torch.empty(0, dtype=torch.long),
                node_last_update=torch.empty(0, 2, dtype=torch.long),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=torch.empty(0, dtype=torch.long),
                edge_last_update=torch.empty(0, 2, dtype=torch.long),
            ),
            3,
            [
                Data(
                    x=torch.empty(0, dtype=torch.long),
                    node_last_update=torch.empty(0, 2, dtype=torch.long),
                    edge_index=torch.empty(2, 0, dtype=torch.long),
                    edge_attr=torch.empty(0, dtype=torch.long),
                    edge_last_update=torch.empty(0, 2, dtype=torch.long),
                ),
                Data(
                    x=torch.empty(0, dtype=torch.long),
                    node_last_update=torch.empty(0, 2, dtype=torch.long),
                    edge_index=torch.empty(2, 0, dtype=torch.long),
                    edge_attr=torch.empty(0, dtype=torch.long),
                    edge_last_update=torch.empty(0, 2, dtype=torch.long),
                ),
                Data(
                    x=torch.empty(0, dtype=torch.long),
                    node_last_update=torch.empty(0, 2, dtype=torch.long),
                    edge_index=torch.empty(2, 0, dtype=torch.long),
                    edge_attr=torch.empty(0, dtype=torch.long),
                    edge_last_update=torch.empty(0, 2, dtype=torch.long),
                ),
            ],
        ),
        (
            Batch(
                batch=torch.tensor([2, 2, 3, 3]),
                x=torch.tensor([3, 4, 5, 6]),
                node_last_update=torch.tensor([[1, 2], [2, 0], [3, 7], [4, 1]]),
                edge_index=torch.tensor([[3], [2]]),
                edge_attr=torch.tensor([2]),
                edge_last_update=torch.tensor([[4, 2]]),
            ),
            4,
            [
                Data(
                    x=torch.empty(0, dtype=torch.long),
                    node_last_update=torch.empty(0, 2, dtype=torch.long),
                    edge_index=torch.empty(2, 0, dtype=torch.long),
                    edge_attr=torch.empty(0, dtype=torch.long),
                    edge_last_update=torch.empty(0, 2, dtype=torch.long),
                ),
                Data(
                    x=torch.empty(0, dtype=torch.long),
                    node_last_update=torch.empty(0, 2, dtype=torch.long),
                    edge_index=torch.empty(2, 0, dtype=torch.long),
                    edge_attr=torch.empty(0, dtype=torch.long),
                    edge_last_update=torch.empty(0, 2, dtype=torch.long),
                ),
                Data(
                    x=torch.tensor([3, 4]),
                    node_last_update=torch.tensor([[1, 2], [2, 0]]),
                    edge_index=torch.empty(2, 0, dtype=torch.long),
                    edge_attr=torch.empty(0, dtype=torch.long),
                    edge_last_update=torch.empty(0, 2, dtype=torch.long),
                ),
                Data(
                    x=torch.tensor([5, 6]),
                    node_last_update=torch.tensor([[3, 7], [4, 1]]),
                    edge_index=torch.tensor([[1], [0]]),
                    edge_attr=torch.tensor([2]),
                    edge_last_update=torch.tensor([[4, 2]]),
                ),
            ],
        ),
        (
            Batch(
                batch=torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3]),
                x=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                node_last_update=torch.tensor(
                    [
                        [0, 2],
                        [1, 0],
                        [2, 3],
                        [3, 1],
                        [4, 6],
                        [5, 9],
                        [6, 1],
                        [7, 3],
                        [8, 2],
                        [1, 2],
                    ]
                ),
                edge_index=torch.tensor([[2, 5, 8], [0, 3, 6]]),
                edge_attr=torch.tensor([1, 2, 3]),
                edge_last_update=torch.tensor([[3, 2], [2, 2], [2, 5]]),
            ),
            4,
            [
                Data(
                    x=torch.tensor([1, 2, 3]),
                    node_last_update=torch.tensor([[0, 2], [1, 0], [2, 3]]),
                    edge_index=torch.tensor([[2], [0]]),
                    edge_attr=torch.tensor([1]),
                    edge_last_update=torch.tensor([[3, 2]]),
                ),
                Data(
                    x=torch.tensor([4, 5, 6]),
                    node_last_update=torch.tensor([[3, 1], [4, 6], [5, 9]]),
                    edge_index=torch.tensor([[2], [0]]),
                    edge_attr=torch.tensor([2]),
                    edge_last_update=torch.tensor([[2, 2]]),
                ),
                Data(
                    x=torch.tensor([7, 8, 9]),
                    node_last_update=torch.tensor([[6, 1], [7, 3], [8, 2]]),
                    edge_index=torch.tensor([[2], [0]]),
                    edge_attr=torch.tensor([3]),
                    edge_last_update=torch.tensor([[2, 5]]),
                ),
                Data(
                    batch=torch.tensor([3]),
                    x=torch.tensor([10]),
                    node_last_update=torch.tensor([[1, 2]]),
                    edge_index=torch.empty(2, 0, dtype=torch.long),
                    edge_attr=torch.empty(0, dtype=torch.long),
                    edge_last_update=torch.empty(0, 2, dtype=torch.long),
                ),
            ],
        ),
    ],
)
def test_batch_to_data_list(batch, batch_size, expected_list):
    data_list = batch_to_data_list(batch, batch_size)
    assert len(data_list) == len(expected_list)
    for data, expected in zip(data_list, expected_list):
        assert data.x.equal(expected.x)
        assert data.node_last_update.equal(expected.node_last_update)
        assert data.edge_index.equal(expected.edge_index)
        assert data.edge_attr.equal(expected.edge_attr)
        assert data.edge_last_update.equal(expected.edge_last_update)


@pytest.mark.parametrize(
    "node_attrs,edge_attrs,expected",
    [
        ([], [], set()),
        (
            [(0, {"label": "player"}), (1, {"label": "living room"})],
            [(0, 1, {"label": "at"})],
            {"player , living room , at"},
        ),
        (
            [
                (0, {"label": "player"}),
                (1, {"label": "living room"}),
                (2, {"label": "exit"}),
            ],
            [(0, 1, {"label": "at"}), (2, 1, {"label": "west of"})],
            {"player , living room , at", "exit , living room , west_of"},
        ),
    ],
)
def test_networkx_to_rdf(node_attrs, edge_attrs, expected):
    graph = nx.DiGraph()
    graph.add_nodes_from(node_attrs)
    graph.add_edges_from(edge_attrs)
    assert networkx_to_rdf(graph) == expected


@pytest.mark.parametrize(
    "node_attrs,edge_attrs,expected",
    [
        (
            [(0, {"label": "player"}), (1, {"label": "living room"})],
            [(0, 1, {"label": "at"})],
            {"player , living room , at"},
        ),
        (
            [
                (0, {"label": "player"}),
                (1, {"label": "living room"}),
                (2, {"label": "exit"}),
            ],
            [(0, 1, {"label": "at"}), (2, 1, {"label": "west of"})],
            {"player , living room , at", "exit , living room , west_of"},
        ),
        (
            [
                (0, {"label": "potato"}),
                (1, {"label": "red"}),
                (2, {"label": "table"}),
            ],
            [(0, 1, {"label": "is"}), (0, 2, {"label": "on"})],
            {"red potato , table , on"},
        ),
        (
            [
                (0, {"label": "potato"}),
                (1, {"label": "red"}),
                (2, {"label": "table"}),
                (3, {"label": "potato"}),
                (4, {"label": "yellow"}),
                (5, {"label": "counter"}),
            ],
            [
                (0, 1, {"label": "is"}),
                (0, 2, {"label": "on"}),
                (3, 4, {"label": "is"}),
                (3, 5, {"label": "on"}),
            ],
            {"red potato , table , on", "yellow potato , counter , on"},
        ),
        (
            [
                (0, {"label": "potato"}),
                (1, {"label": "red"}),
                (2, {"label": "table"}),
                (3, {"label": "potato"}),
                (4, {"label": "yellow"}),
                (5, {"label": "counter"}),
                (6, {"label": "purple"}),
            ],
            [
                (0, 1, {"label": "is"}),
                (0, 2, {"label": "on"}),
                (3, 4, {"label": "is"}),
                (3, 5, {"label": "on"}),
                (3, 6, {"label": "is"}),
            ],
            {"red potato , table , on", "potato , counter , on"},
        ),
    ],
)
def test_networkx_to_rdf_allow_obs_with_same_label(node_attrs, edge_attrs, expected):
    graph = nx.DiGraph()
    graph.add_nodes_from(node_attrs)
    graph.add_edges_from(edge_attrs)
    assert networkx_to_rdf(graph, allow_objs_with_same_label=True) == expected


@pytest.mark.parametrize(
    "rdfs,graph_cmds,expected",
    [
        (
            set(),
            ["add , player , living room , at", "add , exit , living room , east_of"],
            {"player , living room , at", "exit , living room , east_of"},
        ),
        (
            {"player , living room , at", "exit , living room , east_of"},
            [
                "delete , player , living room , at",
                "delete , exit , living room , east_of",
            ],
            set(),
        ),
        (
            set(),
            ["add , player , living room , at", "add , player , living room , at"],
            {"player , living room , at"},
        ),
        (
            {"player , living room , at", "exit , living room , east_of"},
            [
                "delete , exit , living room , east_of",
                "delete , exit , living room , east_of",
            ],
            {"player , living room , at"},
        ),
    ],
)
def test_update_rdf_graph(rdfs, graph_cmds, expected):
    assert update_rdf_graph(rdfs, graph_cmds) == expected
