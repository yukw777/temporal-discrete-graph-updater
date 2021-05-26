from dgu.graph import TextWorldGraph


def test_tw_graph_node():
    g = TextWorldGraph()
    node_id = g.add_node("game0", 0, "n1")
    assert node_id == 0
    assert g.graph.order() == 1
    assert g.graph.nodes[node_id]["game"] == "game0"
    assert g.graph.nodes[node_id]["walkthrough_step"] == 0
    assert g.graph.nodes[node_id]["label"] == "n1"
    assert g.get_node_labels() == ["n1"]

    node_id = g.add_node("game0", 1, "n2")
    assert node_id == 1
    assert g.graph.order() == 2
    assert g.graph.nodes[node_id]["game"] == "game0"
    assert g.graph.nodes[node_id]["walkthrough_step"] == 1
    assert g.graph.nodes[node_id]["label"] == "n2"
    assert g.get_node_labels() == ["n1", "n2"]

    node_id = g.add_node("game1", 0, "n1")
    assert node_id == 2
    assert g.graph.order() == 3
    assert g.graph.nodes[node_id]["game"] == "game1"
    assert g.graph.nodes[node_id]["walkthrough_step"] == 0
    assert g.graph.nodes[node_id]["label"] == "n1"
    assert g.get_node_labels() == ["n1", "n2", "n1"]

    g.remove_node(1)
    assert g.graph.order() == 2
    assert g.get_node_labels() == ["n1", "", "n1"]

    node_id = g.add_node("game2", 0, "n1")
    assert node_id == 1
    assert g.graph.order() == 3
    assert g.graph.nodes[node_id]["game"] == "game2"
    assert g.graph.nodes[node_id]["walkthrough_step"] == 0
    assert g.graph.nodes[node_id]["label"] == "n1"
    assert g.get_node_labels() == ["n1", "n1", "n1"]

    node_id = g.add_node("game2", 0, "n2")
    assert node_id == 3
    assert g.graph.order() == 4
    assert g.graph.nodes[node_id]["game"] == "game2"
    assert g.graph.nodes[node_id]["walkthrough_step"] == 0
    assert g.graph.nodes[node_id]["label"] == "n2"
    assert g.get_node_labels() == ["n1", "n1", "n1", "n2"]


def test_tw_graph_triplet_cmd():
    g = TextWorldGraph()

    events = g.process_triplet_cmd("game0", 0, 0, "add , player , kitchen , in")
    assert events == [
        {
            "type": "node-add",
            "node_id": 0,
            "timestamp": 0,
        },
        {
            "type": "node-add",
            "node_id": 1,
            "timestamp": 0,
        },
        {
            "type": "edge-add",
            "src_id": 0,
            "dst_id": 1,
            "timestamp": 0,
        },
    ]
    assert g.get_node_labels() == ["player", "kitchen"]
    assert set(g.get_edge_labels()) == {(0, 1, "in")}

    # try adding two exits. they should all be added.
    events = g.process_triplet_cmd("game0", 0, 0, "add , exit , kitchen , east_of")
    assert events == [
        {
            "type": "node-add",
            "node_id": 2,
            "timestamp": 0,
        },
        {
            "type": "edge-add",
            "src_id": 2,
            "dst_id": 1,
            "timestamp": 0,
        },
    ]
    assert g.get_node_labels() == ["player", "kitchen", "exit"]
    assert set(g.get_edge_labels()) == {(0, 1, "in"), (2, 1, "east of")}
    events = g.process_triplet_cmd("game0", 0, 0, "add , exit , kitchen , west_of")
    assert events == [
        {
            "type": "node-add",
            "node_id": 3,
            "timestamp": 0,
        },
        {
            "type": "edge-add",
            "src_id": 3,
            "dst_id": 1,
            "timestamp": 0,
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
        {
            "type": "node-add",
            "node_id": 4,
            "timestamp": 0,
        },
        {
            "type": "edge-add",
            "src_id": 4,
            "dst_id": 1,
            "timestamp": 0,
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
        {
            "type": "node-add",
            "node_id": 5,
            "timestamp": 0,
        },
        {
            "type": "edge-add",
            "src_id": 5,
            "dst_id": 1,
            "timestamp": 0,
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
        {
            "type": "node-add",
            "node_id": 6,
            "timestamp": 0,
        },
        {
            "type": "edge-add",
            "src_id": 4,
            "dst_id": 6,
            "timestamp": 0,
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
        {
            "type": "node-add",
            "node_id": 7,
            "timestamp": 0,
        },
        {
            "type": "edge-add",
            "src_id": 5,
            "dst_id": 7,
            "timestamp": 0,
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
        {
            "type": "node-add",
            "node_id": 8,
            "timestamp": 0,
        },
        {
            "type": "node-add",
            "node_id": 9,
            "timestamp": 0,
        },
        {
            "type": "edge-add",
            "src_id": 8,
            "dst_id": 9,
            "timestamp": 0,
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
        {
            "type": "node-add",
            "node_id": 10,
            "timestamp": 1,
        },
        {
            "type": "edge-add",
            "src_id": 10,
            "dst_id": 9,
            "timestamp": 1,
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
        {
            "type": "node-add",
            "node_id": 11,
            "timestamp": 0,
        },
        {
            "type": "node-add",
            "node_id": 12,
            "timestamp": 0,
        },
        {
            "type": "edge-add",
            "src_id": 11,
            "dst_id": 12,
            "timestamp": 0,
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
        {
            "type": "node-add",
            "node_id": 13,
            "timestamp": 2,
        },
        {
            "type": "edge-add",
            "src_id": 13,
            "dst_id": 12,
            "timestamp": 2,
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
            "src_id": 4,
            "dst_id": 6,
            "timestamp": 4,
        },
        {
            "type": "node-delete",
            "node_id": 6,
            "timestamp": 4,
        },
    ]
    assert g.get_node_labels() == [
        "player",
        "kitchen",
        "exit",
        "exit",
        "apple",
        "pear",
        "",
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
            "src_id": 13,
            "dst_id": 12,
            "timestamp": 4,
        },
        {
            "type": "node-delete",
            "node_id": 13,
            "timestamp": 4,
        },
    ]
    assert g.get_node_labels() == [
        "player",
        "kitchen",
        "exit",
        "exit",
        "apple",
        "pear",
        "",
        "cut",
        "player",
        "kitchen",
        "exit",
        "player",
        "living room",
        "",
    ]
    assert set(g.get_edge_labels()) == {
        (0, 1, "in"),
        (2, 1, "east of"),
        (3, 1, "west of"),
        (4, 1, "in"),
        (5, 1, "in"),
        (5, 7, "is"),
        (8, 9, "in"),
        (10, 9, "east of"),
        (11, 12, "in"),
    }

    # add cooked to apple from game0-0
    events = g.process_triplet_cmd("game0", 0, 6, "add , apple , cooked , is")
    assert events == [
        {
            "type": "node-add",
            "node_id": 6,
            "timestamp": 6,
        },
        {
            "type": "edge-add",
            "src_id": 4,
            "dst_id": 6,
            "timestamp": 6,
        },
    ]
    assert g.get_node_labels() == [
        "player",
        "kitchen",
        "exit",
        "exit",
        "apple",
        "pear",
        "cooked",
        "cut",
        "player",
        "kitchen",
        "exit",
        "player",
        "living room",
        "",
    ]
    assert set(g.get_edge_labels()) == {
        (0, 1, "in"),
        (2, 1, "east of"),
        (3, 1, "west of"),
        (4, 1, "in"),
        (5, 1, "in"),
        (5, 7, "is"),
        (8, 9, "in"),
        (10, 9, "east of"),
        (11, 12, "in"),
        (4, 6, "is"),
    }

    # add apple in living room for game1-0
    events = g.process_triplet_cmd("game1", 0, 6, "add , apple , living room , in")
    assert events == [
        {
            "type": "node-add",
            "node_id": 13,
            "timestamp": 6,
        },
        {
            "type": "edge-add",
            "src_id": 13,
            "dst_id": 12,
            "timestamp": 6,
        },
    ]
    assert g.get_node_labels() == [
        "player",
        "kitchen",
        "exit",
        "exit",
        "apple",
        "pear",
        "cooked",
        "cut",
        "player",
        "kitchen",
        "exit",
        "player",
        "living room",
        "apple",
    ]
    assert set(g.get_edge_labels()) == {
        (0, 1, "in"),
        (2, 1, "east of"),
        (3, 1, "west of"),
        (4, 1, "in"),
        (5, 1, "in"),
        (5, 7, "is"),
        (8, 9, "in"),
        (10, 9, "east of"),
        (11, 12, "in"),
        (4, 6, "is"),
        (13, 12, "in"),
    }
