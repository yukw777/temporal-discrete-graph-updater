import pytest

from dgu.graph import TextWorldGraph


def test_tw_graph_node():
    g = TextWorldGraph()
    node_id = g.add_node("n1", game="game0", walkthrough_step=0)
    assert node_id == 0
    assert g._graph.order() == 1
    assert g._graph.nodes[node_id]["game"] == "game0"
    assert g._graph.nodes[node_id]["walkthrough_step"] == 0
    assert g._graph.nodes[node_id]["label"] == "n1"
    assert not g._graph.nodes[node_id]["removed"]
    assert g.get_node_labels() == ["n1"]

    node_id = g.add_node("n2", game="game0", walkthrough_step=1)
    assert node_id == 1
    assert g._graph.order() == 2
    assert g._graph.nodes[node_id]["game"] == "game0"
    assert g._graph.nodes[node_id]["walkthrough_step"] == 1
    assert g._graph.nodes[node_id]["label"] == "n2"
    assert not g._graph.nodes[node_id]["removed"]
    assert g.get_node_labels() == ["n1", "n2"]

    node_id = g.add_node("n1", game="game1", walkthrough_step=0)
    assert node_id == 2
    assert g._graph.order() == 3
    assert g._graph.nodes[node_id]["game"] == "game1"
    assert g._graph.nodes[node_id]["walkthrough_step"] == 0
    assert g._graph.nodes[node_id]["label"] == "n1"
    assert not g._graph.nodes[node_id]["removed"]
    assert g.get_node_labels() == ["n1", "n2", "n1"]

    g.remove_node(1)
    assert g._graph.order() == 3
    assert g._graph.nodes[1]["game"] == "game0"
    assert g._graph.nodes[1]["walkthrough_step"] == 1
    assert g._graph.nodes[1]["label"] == "n2"
    assert g._graph.nodes[1]["removed"]
    assert g.get_node_labels() == ["n1", "n2", "n1"]

    node_id = g.add_node("n1", game="game2", walkthrough_step=0)
    assert node_id == 3
    assert g._graph.order() == 4
    assert g._graph.nodes[node_id]["game"] == "game2"
    assert g._graph.nodes[node_id]["walkthrough_step"] == 0
    assert g._graph.nodes[node_id]["label"] == "n1"
    assert g.get_node_labels() == ["n1", "n2", "n1", "n1"]

    node_id = g.add_node("n2", game="game2", walkthrough_step=0)
    assert node_id == 4
    assert g._graph.order() == 5
    assert g._graph.nodes[node_id]["game"] == "game2"
    assert g._graph.nodes[node_id]["walkthrough_step"] == 0
    assert g._graph.nodes[node_id]["label"] == "n2"
    assert g.get_node_labels() == ["n1", "n2", "n1", "n1", "n2"]


def test_tw_graph_edge():
    g = TextWorldGraph()
    n1_id = g.add_node("n1", game="game0", walkthrough_step=0)
    n2_id = g.add_node("n2", game="game0", walkthrough_step=1)
    e1_id = g.add_edge(n1_id, n2_id, "e1", game="game0", walkthrough_step=1)
    assert e1_id == 0
    assert g._graph.number_of_edges() == 1
    assert g._graph[n1_id][n2_id]["id"] == e1_id
    assert g._graph[n1_id][n2_id]["game"] == "game0"
    assert g._graph[n1_id][n2_id]["walkthrough_step"] == 1
    assert g._graph[n1_id][n2_id]["label"] == "e1"
    assert not g._graph[n1_id][n2_id]["removed"]

    n3_id = g.add_node("n3", game="game0", walkthrough_step=2)
    e2_id = g.add_edge(n2_id, n3_id, "e2", game="game0", walkthrough_step=2)
    assert e2_id == 1
    assert g._graph.number_of_edges() == 2
    assert g._graph[n2_id][n3_id]["id"] == e2_id
    assert g._graph[n2_id][n3_id]["game"] == "game0"
    assert g._graph[n2_id][n3_id]["walkthrough_step"] == 2
    assert g._graph[n2_id][n3_id]["label"] == "e2"
    assert not g._graph[n2_id][n3_id]["removed"]

    n4_id = g.add_node("n4", game="game1", walkthrough_step=0)
    n5_id = g.add_node("n5", game="game1", walkthrough_step=1)
    e3_id = g.add_edge(n4_id, n5_id, "e3", game="game1", walkthrough_step=1)
    assert e3_id == 2
    assert g._graph.number_of_edges() == 3
    assert g._graph[n4_id][n5_id]["id"] == e3_id
    assert g._graph[n4_id][n5_id]["game"] == "game1"
    assert g._graph[n4_id][n5_id]["walkthrough_step"] == 1
    assert g._graph[n4_id][n5_id]["label"] == "e3"
    assert not g._graph[n4_id][n5_id]["removed"]

    removed_e1_id = g.remove_edge(n1_id, n2_id)
    assert e1_id == removed_e1_id
    assert g._graph.number_of_edges() == 3
    assert g._graph[n1_id][n2_id]["removed"]

    removed_e2_id = g.remove_edge(n2_id, n3_id)
    assert e2_id == removed_e2_id
    assert g._graph.number_of_edges() == 3
    assert g._graph[n2_id][n3_id]["removed"]

    readded_e2_id = g.add_edge(
        n2_id, n3_id, "readded_e2", game="game0", walkthrough_step=2
    )
    assert readded_e2_id == e2_id
    assert g._graph.number_of_edges() == 3
    assert g._graph[n2_id][n3_id]["id"] == readded_e2_id
    assert g._graph[n2_id][n3_id]["game"] == "game0"
    assert g._graph[n2_id][n3_id]["walkthrough_step"] == 2
    assert g._graph[n2_id][n3_id]["label"] == "readded_e2"
    assert not g._graph[n2_id][n3_id]["removed"]

    n6_id = g.add_node("n6", game="game1", walkthrough_step=2)
    e4_id = g.add_edge(n5_id, n6_id, "e4", game="game1", walkthrough_step=2)
    assert e4_id == 3
    assert g._graph.number_of_edges() == 4
    assert g._graph[n5_id][n6_id]["id"] == e4_id
    assert g._graph[n5_id][n6_id]["game"] == "game1"
    assert g._graph[n5_id][n6_id]["walkthrough_step"] == 2
    assert g._graph[n5_id][n6_id]["label"] == "e4"
    assert not g._graph[n5_id][n6_id]["removed"]


def test_tw_graph_triplet_cmd():
    g = TextWorldGraph()

    events = g.process_triplet_cmd("game0", 0, 0, "add , player , kitchen , in")
    assert events == [
        {"type": "node-add", "node_id": 0, "timestamp": 0, "label": "player"},
        {"type": "node-add", "node_id": 1, "timestamp": 0, "label": "kitchen"},
        {
            "type": "edge-add",
            "edge_id": 0,
            "src_id": 0,
            "dst_id": 1,
            "timestamp": 0,
            "label": "in",
        },
    ]
    assert g.get_node_labels() == ["player", "kitchen"]
    assert set(g.get_edge_labels()) == {(0, 1, "in")}

    # try adding two exits. they should all be added.
    events = g.process_triplet_cmd("game0", 0, 0, "add , exit , kitchen , east_of")
    assert events == [
        {"type": "node-add", "node_id": 2, "timestamp": 0, "label": "exit"},
        {
            "type": "edge-add",
            "edge_id": 1,
            "src_id": 2,
            "dst_id": 1,
            "timestamp": 0,
            "label": "east of",
        },
    ]
    assert g.get_node_labels() == ["player", "kitchen", "exit"]
    assert set(g.get_edge_labels()) == {(0, 1, "in"), (2, 1, "east of")}
    events = g.process_triplet_cmd("game0", 0, 0, "add , exit , kitchen , west_of")
    assert events == [
        {"type": "node-add", "node_id": 3, "timestamp": 0, "label": "exit"},
        {
            "type": "edge-add",
            "edge_id": 2,
            "src_id": 3,
            "dst_id": 1,
            "timestamp": 0,
            "label": "west of",
        },
    ]
    assert g.get_node_labels() == ["player", "kitchen", "exit", "exit"]
    assert set(g.get_edge_labels()) == {
        (0, 1, "in"),
        (2, 1, "east of"),
        (3, 1, "west of"),
    }

    events = g.process_triplet_cmd("game0", 0, 0, "add , apple , kitchen , in")
    assert events == [
        {"type": "node-add", "node_id": 4, "timestamp": 0, "label": "apple"},
        {
            "type": "edge-add",
            "edge_id": 3,
            "src_id": 4,
            "dst_id": 1,
            "timestamp": 0,
            "label": "in",
        },
    ]
    assert g.get_node_labels() == ["player", "kitchen", "exit", "exit", "apple"]
    assert set(g.get_edge_labels()) == {
        (0, 1, "in"),
        (2, 1, "east of"),
        (3, 1, "west of"),
        (4, 1, "in"),
    }
    events = g.process_triplet_cmd("game0", 0, 0, "add , pear , kitchen , in")
    assert events == [
        {"type": "node-add", "node_id": 5, "timestamp": 0, "label": "pear"},
        {
            "type": "edge-add",
            "edge_id": 4,
            "src_id": 5,
            "dst_id": 1,
            "timestamp": 0,
            "label": "in",
        },
    ]
    assert g.get_node_labels() == ["player", "kitchen", "exit", "exit", "apple", "pear"]
    assert set(g.get_edge_labels()) == {
        (0, 1, "in"),
        (2, 1, "east of"),
        (3, 1, "west of"),
        (4, 1, "in"),
        (5, 1, "in"),
    }

    # try adding "cut" to apple and pear, they should both be added.
    events = g.process_triplet_cmd("game0", 0, 0, "add , apple , cut , is")
    assert events == [
        {"type": "node-add", "node_id": 6, "timestamp": 0, "label": "cut"},
        {
            "type": "edge-add",
            "edge_id": 5,
            "src_id": 4,
            "dst_id": 6,
            "timestamp": 0,
            "label": "is",
        },
    ]
    assert g.get_node_labels() == [
        "player",
        "kitchen",
        "exit",
        "exit",
        "apple",
        "pear",
        "cut",
    ]
    assert set(g.get_edge_labels()) == {
        (0, 1, "in"),
        (2, 1, "east of"),
        (3, 1, "west of"),
        (4, 1, "in"),
        (5, 1, "in"),
        (4, 6, "is"),
    }

    events = g.process_triplet_cmd("game0", 0, 0, "add , pear , cut , is")
    assert events == [
        {"type": "node-add", "node_id": 7, "timestamp": 0, "label": "cut"},
        {
            "type": "edge-add",
            "edge_id": 6,
            "src_id": 5,
            "dst_id": 7,
            "timestamp": 0,
            "label": "is",
        },
    ]
    assert g.get_node_labels() == [
        "player",
        "kitchen",
        "exit",
        "exit",
        "apple",
        "pear",
        "cut",
        "cut",
    ]
    assert set(g.get_edge_labels()) == {
        (0, 1, "in"),
        (2, 1, "east of"),
        (3, 1, "west of"),
        (4, 1, "in"),
        (4, 6, "is"),
        (5, 1, "in"),
        (5, 7, "is"),
    }

    # same game, same walkthrough step, so shouldn't be added
    events = g.process_triplet_cmd("game0", 0, 0, "add , player , kitchen , in")
    assert events == []
    events = g.process_triplet_cmd("game0", 0, 1, "add , exit , kitchen , east_of")
    assert events == []
    events = g.process_triplet_cmd("game0", 0, 2, "add , exit , kitchen , west_of")
    assert events == []
    events = g.process_triplet_cmd("game0", 0, 3, "add , apple , kitchen , in")
    assert events == []
    events = g.process_triplet_cmd("game0", 0, 3, "add , pear , kitchen , in")
    assert events == []
    events = g.process_triplet_cmd("game0", 0, 4, "add , apple , cut , is")
    assert events == []
    events = g.process_triplet_cmd("game0", 0, 4, "add , pear , cut , is")
    assert events == []
    assert g.get_node_labels() == [
        "player",
        "kitchen",
        "exit",
        "exit",
        "apple",
        "pear",
        "cut",
        "cut",
    ]
    assert set(g.get_edge_labels()) == {
        (0, 1, "in"),
        (2, 1, "east of"),
        (3, 1, "west of"),
        (4, 1, "in"),
        (4, 6, "is"),
        (5, 1, "in"),
        (5, 7, "is"),
    }

    # same game, different walkthrough step, so should be added
    events = g.process_triplet_cmd("game0", 1, 0, "add , player , kitchen , in")
    assert events == [
        {"type": "node-add", "node_id": 8, "timestamp": 0, "label": "player"},
        {"type": "node-add", "node_id": 9, "timestamp": 0, "label": "kitchen"},
        {
            "type": "edge-add",
            "edge_id": 7,
            "src_id": 8,
            "dst_id": 9,
            "timestamp": 0,
            "label": "in",
        },
    ]
    assert g.get_node_labels() == [
        "player",
        "kitchen",
        "exit",
        "exit",
        "apple",
        "pear",
        "cut",
        "cut",
        "player",
        "kitchen",
    ]
    assert set(g.get_edge_labels()) == {
        (0, 1, "in"),
        (2, 1, "east of"),
        (3, 1, "west of"),
        (4, 1, "in"),
        (4, 6, "is"),
        (5, 1, "in"),
        (5, 7, "is"),
        (8, 9, "in"),
    }
    events = g.process_triplet_cmd("game0", 1, 1, "add , exit , kitchen , east_of")
    assert events == [
        {"type": "node-add", "node_id": 10, "timestamp": 1, "label": "exit"},
        {
            "type": "edge-add",
            "edge_id": 8,
            "src_id": 10,
            "dst_id": 9,
            "timestamp": 1,
            "label": "east of",
        },
    ]
    assert g.get_node_labels() == [
        "player",
        "kitchen",
        "exit",
        "exit",
        "apple",
        "pear",
        "cut",
        "cut",
        "player",
        "kitchen",
        "exit",
    ]
    assert set(g.get_edge_labels()) == {
        (0, 1, "in"),
        (2, 1, "east of"),
        (3, 1, "west of"),
        (4, 1, "in"),
        (4, 6, "is"),
        (5, 1, "in"),
        (5, 7, "is"),
        (8, 9, "in"),
        (10, 9, "east of"),
    }

    # different game, same walkthrough step, so should be added
    events = g.process_triplet_cmd("game1", 0, 0, "add , player , living room , in")
    assert events == [
        {"type": "node-add", "node_id": 11, "timestamp": 0, "label": "player"},
        {"type": "node-add", "node_id": 12, "timestamp": 0, "label": "living room"},
        {
            "type": "edge-add",
            "edge_id": 9,
            "src_id": 11,
            "dst_id": 12,
            "timestamp": 0,
            "label": "in",
        },
    ]
    assert g.get_node_labels() == [
        "player",
        "kitchen",
        "exit",
        "exit",
        "apple",
        "pear",
        "cut",
        "cut",
        "player",
        "kitchen",
        "exit",
        "player",
        "living room",
    ]
    assert set(g.get_edge_labels()) == {
        (0, 1, "in"),
        (2, 1, "east of"),
        (3, 1, "west of"),
        (4, 1, "in"),
        (4, 6, "is"),
        (5, 1, "in"),
        (5, 7, "is"),
        (8, 9, "in"),
        (10, 9, "east of"),
        (11, 12, "in"),
    }
    events = g.process_triplet_cmd("game1", 0, 2, "add , exit , living room , east_of")
    assert events == [
        {"type": "node-add", "node_id": 13, "timestamp": 2, "label": "exit"},
        {
            "type": "edge-add",
            "edge_id": 10,
            "src_id": 13,
            "dst_id": 12,
            "timestamp": 2,
            "label": "east of",
        },
    ]
    assert g.get_node_labels() == [
        "player",
        "kitchen",
        "exit",
        "exit",
        "apple",
        "pear",
        "cut",
        "cut",
        "player",
        "kitchen",
        "exit",
        "player",
        "living room",
        "exit",
    ]
    assert set(g.get_edge_labels()) == {
        (0, 1, "in"),
        (2, 1, "east of"),
        (3, 1, "west of"),
        (4, 1, "in"),
        (4, 6, "is"),
        (5, 1, "in"),
        (5, 7, "is"),
        (8, 9, "in"),
        (10, 9, "east of"),
        (11, 12, "in"),
        (13, 12, "east of"),
    }

    # delete apple from game0-0, should delete cut
    events = g.process_triplet_cmd("game0", 0, 4, "delete , apple , cut , is")
    assert events == [
        {
            "type": "edge-delete",
            "edge_id": 5,
            "src_id": 4,
            "dst_id": 6,
            "timestamp": 4,
            "label": "is",
        },
        {"type": "node-delete", "node_id": 6, "timestamp": 4, "label": "cut"},
    ]
    assert g.get_node_labels() == [
        "player",
        "kitchen",
        "exit",
        "exit",
        "apple",
        "pear",
        "cut",
        "cut",
        "player",
        "kitchen",
        "exit",
        "player",
        "living room",
        "exit",
    ]
    assert set(g.get_edge_labels()) == {
        (0, 1, "in"),
        (2, 1, "east of"),
        (3, 1, "west of"),
        (4, 1, "in"),
        (4, 6, "is"),
        (5, 1, "in"),
        (5, 7, "is"),
        (8, 9, "in"),
        (10, 9, "east of"),
        (11, 12, "in"),
        (13, 12, "east of"),
    }

    # delete exit from game1-0
    events = g.process_triplet_cmd(
        "game1", 0, 4, "delete , exit , living room , east_of"
    )
    assert events == [
        {
            "type": "edge-delete",
            "edge_id": 10,
            "src_id": 13,
            "dst_id": 12,
            "timestamp": 4,
            "label": "east of",
        },
        {"type": "node-delete", "node_id": 13, "timestamp": 4, "label": "exit"},
    ]
    assert g.get_node_labels() == [
        "player",
        "kitchen",
        "exit",
        "exit",
        "apple",
        "pear",
        "cut",
        "cut",
        "player",
        "kitchen",
        "exit",
        "player",
        "living room",
        "exit",
    ]
    assert set(g.get_edge_labels()) == {
        (0, 1, "in"),
        (2, 1, "east of"),
        (3, 1, "west of"),
        (4, 1, "in"),
        (4, 6, "is"),
        (5, 1, "in"),
        (5, 7, "is"),
        (8, 9, "in"),
        (10, 9, "east of"),
        (11, 12, "in"),
        (13, 12, "east of"),
    }

    # add cooked to apple from game0-0
    events = g.process_triplet_cmd("game0", 0, 6, "add , apple , cooked , is")
    assert events == [
        {"type": "node-add", "node_id": 14, "timestamp": 6, "label": "cooked"},
        {
            "type": "edge-add",
            "edge_id": 11,
            "src_id": 4,
            "dst_id": 14,
            "timestamp": 6,
            "label": "is",
        },
    ]
    assert g.get_node_labels() == [
        "player",
        "kitchen",
        "exit",
        "exit",
        "apple",
        "pear",
        "cut",
        "cut",
        "player",
        "kitchen",
        "exit",
        "player",
        "living room",
        "exit",
        "cooked",
    ]
    assert set(g.get_edge_labels()) == {
        (0, 1, "in"),
        (2, 1, "east of"),
        (3, 1, "west of"),
        (4, 1, "in"),
        (4, 6, "is"),
        (4, 14, "is"),
        (5, 1, "in"),
        (5, 7, "is"),
        (8, 9, "in"),
        (10, 9, "east of"),
        (11, 12, "in"),
        (13, 12, "east of"),
    }

    # add apple in living room for game1-0
    events = g.process_triplet_cmd("game1", 0, 6, "add , apple , living room , in")
    assert events == [
        {"type": "node-add", "node_id": 15, "timestamp": 6, "label": "apple"},
        {
            "type": "edge-add",
            "edge_id": 12,
            "src_id": 15,
            "dst_id": 12,
            "timestamp": 6,
            "label": "in",
        },
    ]
    assert g.get_node_labels() == [
        "player",
        "kitchen",
        "exit",
        "exit",
        "apple",
        "pear",
        "cut",
        "cut",
        "player",
        "kitchen",
        "exit",
        "player",
        "living room",
        "exit",
        "cooked",
        "apple",
    ]
    assert set(g.get_edge_labels()) == {
        (0, 1, "in"),
        (2, 1, "east of"),
        (3, 1, "west of"),
        (4, 1, "in"),
        (4, 6, "is"),
        (4, 14, "is"),
        (5, 1, "in"),
        (5, 7, "is"),
        (8, 9, "in"),
        (10, 9, "east of"),
        (11, 12, "in"),
        (13, 12, "east of"),
        (15, 12, "in"),
    }


def test_tw_graph_triplet_cmd_delete():
    # make sure we handle deletion correctly by deleting a node and adding back in
    # set up a graph
    g = TextWorldGraph()
    g.process_triplet_cmd("game0", 0, 0, "add , apple , kitchen , in")
    g.process_triplet_cmd("game0", 0, 0, "add , exit , kitchen , east_of")
    g.process_triplet_cmd("game0", 0, 0, "add , fridge , kitchen , in")
    g.process_triplet_cmd("game0", 0, 0, "add , fridge , closed , is")
    g.process_triplet_cmd("game0", 0, 0, "add , kitchen , living room , east_of")

    # regular source node
    assert g.process_triplet_cmd("game0", 0, 1, "delete , apple , kitchen , in") == [
        {
            "type": "edge-delete",
            "edge_id": 0,
            "src_id": 0,
            "dst_id": 1,
            "timestamp": 1,
            "label": "in",
        },
        {"type": "node-delete", "node_id": 0, "timestamp": 1, "label": "apple"},
    ]
    assert g.get_node_labels() == [
        "apple",
        "kitchen",
        "exit",
        "fridge",
        "closed",
        "living room",
    ]
    assert set(g.get_edge_labels()) == {
        (0, 1, "in"),
        (1, 5, "east of"),
        (2, 1, "east of"),
        (3, 1, "in"),
        (3, 4, "is"),
    }
    assert g.process_triplet_cmd("game0", 0, 2, "add , apple , kitchen , in") == [
        {"type": "node-add", "node_id": 6, "timestamp": 2, "label": "apple"},
        {
            "type": "edge-add",
            "edge_id": 5,
            "src_id": 6,
            "dst_id": 1,
            "timestamp": 2,
            "label": "in",
        },
    ]
    assert g.get_node_labels() == [
        "apple",
        "kitchen",
        "exit",
        "fridge",
        "closed",
        "living room",
        "apple",
    ]
    assert set(g.get_edge_labels()) == {
        (0, 1, "in"),
        (1, 5, "east of"),
        (2, 1, "east of"),
        (3, 1, "in"),
        (3, 4, "is"),
        (6, 1, "in"),
    }

    # regular destination node
    assert g.process_triplet_cmd(
        "game0", 0, 3, "delete , kitchen , living room , east_of"
    ) == [
        {
            "type": "edge-delete",
            "edge_id": 4,
            "src_id": 1,
            "dst_id": 5,
            "timestamp": 3,
            "label": "east of",
        },
        {"type": "node-delete", "node_id": 5, "timestamp": 3, "label": "living room"},
    ]
    assert g.get_node_labels() == [
        "apple",
        "kitchen",
        "exit",
        "fridge",
        "closed",
        "living room",
        "apple",
    ]
    assert set(g.get_edge_labels()) == {
        (0, 1, "in"),
        (1, 5, "east of"),
        (2, 1, "east of"),
        (3, 1, "in"),
        (3, 4, "is"),
        (6, 1, "in"),
    }
    assert g.process_triplet_cmd(
        "game0", 0, 4, "add , kitchen , living room , east_of"
    ) == [
        {"type": "node-add", "node_id": 7, "timestamp": 4, "label": "living room"},
        {
            "type": "edge-add",
            "edge_id": 6,
            "src_id": 1,
            "dst_id": 7,
            "timestamp": 4,
            "label": "east of",
        },
    ]
    assert g.get_node_labels() == [
        "apple",
        "kitchen",
        "exit",
        "fridge",
        "closed",
        "living room",
        "apple",
        "living room",
    ]
    assert set(g.get_edge_labels()) == {
        (0, 1, "in"),
        (1, 5, "east of"),
        (1, 7, "east of"),
        (2, 1, "east of"),
        (3, 1, "in"),
        (3, 4, "is"),
        (6, 1, "in"),
    }

    # is source node
    assert g.process_triplet_cmd("game0", 0, 5, "delete , fridge , closed , is") == [
        {
            "type": "edge-delete",
            "edge_id": 3,
            "src_id": 3,
            "dst_id": 4,
            "timestamp": 5,
            "label": "is",
        },
        {"type": "node-delete", "node_id": 4, "timestamp": 5, "label": "closed"},
    ]
    assert g.get_node_labels() == [
        "apple",
        "kitchen",
        "exit",
        "fridge",
        "closed",
        "living room",
        "apple",
        "living room",
    ]
    assert set(g.get_edge_labels()) == {
        (0, 1, "in"),
        (1, 5, "east of"),
        (1, 7, "east of"),
        (2, 1, "east of"),
        (3, 1, "in"),
        (3, 4, "is"),
        (6, 1, "in"),
    }
    assert g.process_triplet_cmd("game0", 0, 6, "add , fridge , closed , is") == [
        {"type": "node-add", "node_id": 8, "timestamp": 6, "label": "closed"},
        {
            "type": "edge-add",
            "edge_id": 7,
            "src_id": 3,
            "dst_id": 8,
            "timestamp": 6,
            "label": "is",
        },
    ]
    assert g.get_node_labels() == [
        "apple",
        "kitchen",
        "exit",
        "fridge",
        "closed",
        "living room",
        "apple",
        "living room",
        "closed",
    ]
    assert set(g.get_edge_labels()) == {
        (0, 1, "in"),
        (1, 5, "east of"),
        (1, 7, "east of"),
        (2, 1, "east of"),
        (3, 1, "in"),
        (3, 4, "is"),
        (3, 8, "is"),
        (6, 1, "in"),
    }

    # exit source node
    assert g.process_triplet_cmd(
        "game0", 0, 7, "delete , exit , kitchen , east_of"
    ) == [
        {
            "type": "edge-delete",
            "edge_id": 1,
            "src_id": 2,
            "dst_id": 1,
            "timestamp": 7,
            "label": "east of",
        },
        {"type": "node-delete", "node_id": 2, "timestamp": 7, "label": "exit"},
    ]
    assert g.get_node_labels() == [
        "apple",
        "kitchen",
        "exit",
        "fridge",
        "closed",
        "living room",
        "apple",
        "living room",
        "closed",
    ]
    assert set(g.get_edge_labels()) == {
        (0, 1, "in"),
        (1, 5, "east of"),
        (1, 7, "east of"),
        (2, 1, "east of"),
        (3, 1, "in"),
        (3, 4, "is"),
        (3, 8, "is"),
        (6, 1, "in"),
    }
    assert g.process_triplet_cmd("game0", 0, 8, "add , exit , kitchen , east_of") == [
        {"type": "node-add", "node_id": 9, "timestamp": 8, "label": "exit"},
        {
            "type": "edge-add",
            "edge_id": 8,
            "src_id": 9,
            "dst_id": 1,
            "timestamp": 8,
            "label": "east of",
        },
    ]
    assert g.get_node_labels() == [
        "apple",
        "kitchen",
        "exit",
        "fridge",
        "closed",
        "living room",
        "apple",
        "living room",
        "closed",
        "exit",
    ]
    assert set(g.get_edge_labels()) == {
        (0, 1, "in"),
        (1, 5, "east of"),
        (1, 7, "east of"),
        (2, 1, "east of"),
        (3, 1, "in"),
        (3, 4, "is"),
        (3, 8, "is"),
        (6, 1, "in"),
        (9, 1, "east of"),
    }


@pytest.mark.parametrize(
    "batch,expected",
    [
        (
            [
                {
                    "game": "g0",
                    "walkthrough_step": 0,
                    "event_seq": [
                        {"type": "node-add", "label": "n0"},
                        {"type": "node-add", "label": "n1"},
                        {"type": "edge-add", "src_id": 0, "dst_id": 1, "label": "e0"},
                    ],
                }
            ],
            {("g0", 0): ({0, 1}, [0], [(0, 1)])},
        ),
        (
            [
                {
                    "game": "g0",
                    "walkthrough_step": 0,
                    "event_seq": [
                        {"type": "node-add", "label": "n0"},
                        {"type": "node-add", "label": "n1"},
                        {"type": "node-add", "label": "n2"},
                        {"type": "node-add", "label": "n3"},
                        {"type": "edge-add", "src_id": 0, "dst_id": 1, "label": "e0"},
                        {"type": "edge-add", "src_id": 0, "dst_id": 2, "label": "e1"},
                        {
                            "type": "edge-delete",
                            "src_id": 0,
                            "dst_id": 1,
                            "label": "e0",
                        },
                        {"type": "edge-add", "src_id": 0, "dst_id": 3, "label": "e2"},
                    ],
                }
            ],
            {("g0", 0): ({0, 1, 2, 3}, [0, 1, 2], [(0, 1), (0, 2), (0, 3)])},
        ),
        (
            [
                {
                    "game": "g0",
                    "walkthrough_step": 0,
                    "event_seq": [
                        {"type": "node-add", "label": "n0"},
                        {"type": "node-add", "label": "n1"},
                        {"type": "node-add", "label": "n2"},
                        {"type": "edge-add", "src_id": 0, "dst_id": 1, "label": "e0"},
                        {"type": "edge-add", "src_id": 0, "dst_id": 2, "label": "e1"},
                        {
                            "type": "edge-delete",
                            "src_id": 0,
                            "dst_id": 2,
                            "label": "e1",
                        },
                        {
                            "type": "edge-delete",
                            "src_id": 0,
                            "dst_id": 1,
                            "label": "e0",
                        },
                        {"type": "edge-add", "src_id": 0, "dst_id": 1, "label": "e0"},
                    ],
                }
            ],
            {("g0", 0): ({0, 1, 2}, [0, 1], [(0, 1), (0, 2)])},
        ),
        (
            [
                {
                    "game": "g0",
                    "walkthrough_step": 0,
                    "event_seq": [
                        {"type": "node-add", "label": "n0"},
                        {"type": "node-add", "label": "n1"},
                        {"type": "edge-add", "src_id": 0, "dst_id": 1, "label": "e0"},
                    ],
                },
                {
                    "game": "g1",
                    "walkthrough_step": 0,
                    "event_seq": [
                        {"type": "node-add", "label": "n0"},
                        {"type": "node-add", "label": "n1"},
                        {"type": "node-add", "label": "n2"},
                        {"type": "node-add", "label": "n3"},
                        {"type": "edge-add", "src_id": 2, "dst_id": 3, "label": "e0"},
                        {"type": "edge-add", "src_id": 2, "dst_id": 4, "label": "e1"},
                        {
                            "type": "edge-delete",
                            "src_id": 2,
                            "dst_id": 3,
                            "label": "e0",
                        },
                        {"type": "edge-add", "src_id": 2, "dst_id": 5, "label": "e2"},
                    ],
                },
                {
                    "game": "g1",
                    "walkthrough_step": 1,
                    "event_seq": [
                        {"type": "node-add", "label": "n0"},
                        {"type": "node-add", "label": "n1"},
                        {"type": "node-add", "label": "n2"},
                        {"type": "edge-add", "src_id": 6, "dst_id": 7, "label": "e0"},
                        {"type": "edge-add", "src_id": 6, "dst_id": 8, "label": "e1"},
                        {
                            "type": "edge-delete",
                            "src_id": 6,
                            "dst_id": 8,
                            "label": "e1",
                        },
                        {
                            "type": "edge-delete",
                            "src_id": 6,
                            "dst_id": 7,
                            "label": "e0",
                        },
                        {"type": "edge-add", "src_id": 6, "dst_id": 7, "label": "e0"},
                    ],
                },
            ],
            {
                ("g0", 0): ({0, 1}, [0], [(0, 1)]),
                ("g1", 0): ({2, 3, 4, 5}, [1, 2, 3], [(2, 3), (2, 4), (2, 5)]),
                ("g1", 1): ({6, 7, 8}, [4, 5], [(6, 7), (6, 8)]),
            },
        ),
    ],
)
def test_tw_graph_get_subgraph(batch, expected):
    g = TextWorldGraph()
    for ex in batch:
        g.process_events(
            ex["event_seq"], game=ex["game"], walkthrough_step=ex["walkthrough_step"]
        )
    for gw, answer in expected.items():
        assert g.get_subgraph({gw}) == answer


@pytest.mark.parametrize(
    "events,expected_node_labels,expected_edge_labels",
    [
        (
            [{"type": "node-add", "node_id": 0, "timestamp": 5, "label": "n1"}],
            ["n1"],
            set(),
        ),
        (
            [
                {"type": "node-add", "node_id": 0, "timestamp": 5, "label": "n1"},
                {"type": "node-delete", "node_id": 0, "timestamp": 5, "label": "n1"},
            ],
            ["n1"],
            set(),
        ),
        (
            [
                {"type": "node-add", "node_id": 2, "timestamp": 5, "label": "n1"},
                {"type": "node-add", "node_id": 0, "timestamp": 5, "label": "n1"},
                {"type": "node-delete", "node_id": 0, "timestamp": 5, "label": "n1"},
            ],
            ["n1", "n1"],
            set(),
        ),
        (
            [
                {"type": "node-add", "node_id": 1, "timestamp": 5, "label": "n1"},
                {"type": "node-add", "node_id": 0, "timestamp": 5, "label": "n2"},
                {
                    "type": "edge-add",
                    "src_id": 0,
                    "dst_id": 1,
                    "timestamp": 5,
                    "label": "e1",
                },
            ],
            ["n1", "n2"],
            {(0, 1, "e1")},
        ),
        (
            [
                {"type": "node-add", "node_id": 1, "timestamp": 5, "label": "n1"},
                {"type": "node-add", "node_id": 0, "timestamp": 5, "label": "n2"},
                {
                    "type": "edge-add",
                    "src_id": 0,
                    "dst_id": 1,
                    "timestamp": 5,
                    "label": "e1",
                },
                {
                    "type": "edge-delete",
                    "src_id": 0,
                    "dst_id": 1,
                    "timestamp": 6,
                    "label": "e1",
                },
                {
                    "type": "edge-add",
                    "src_id": 1,
                    "dst_id": 0,
                    "timestamp": 7,
                    "label": "e2",
                },
            ],
            ["n1", "n2"],
            {(0, 1, "e1"), (1, 0, "e2")},
        ),
    ],
)
def test_process_event(events, expected_node_labels, expected_edge_labels):
    g = TextWorldGraph()
    attr = "hello"
    g.process_events(events, attr=attr)
    assert g.get_node_labels() == expected_node_labels
    assert set(g.get_edge_labels()) == expected_edge_labels
    for _, data in g._graph.nodes.data():
        assert data["attr"] == attr
    for _, _, data in g._graph.edges.data():
        assert data["attr"] == attr
