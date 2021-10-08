import pytest

from dgu.graph import Node, IsDstNode, ExitNode, process_triplet_cmd

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
                        IsDstNode("cooked", "steak"): {"label": "is", "last_update": 1}
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
                {"type": "edge-delete", "src_id": 0, "dst_id": 1, "label": "in"},
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
                },
                {"type": "node-delete", "node_id": 1, "label": "exit"},
            ],
        ),
        # is destination node deletion
        (
            EqualityDiGraph(
                {
                    Node("steak"): {
                        IsDstNode("cooked", "steak"): {"label": "is", "last_update": 3},
                        IsDstNode("delicious", "steak"): {
                            "label": "is",
                            "last_update": 3,
                        },
                    }
                }
            ),
            4,
            "delete , steak , delicious , is",
            [
                {"type": "edge-delete", "src_id": 0, "dst_id": 2, "label": "is"},
                {
                    "type": "node-delete",
                    "node_id": 2,
                    "label": "delicious",
                },
            ],
        ),
    ],
)
def test_process_triplet_cmd(graph, timestamp, cmd, expected):
    assert process_triplet_cmd(graph, timestamp, cmd) == expected
