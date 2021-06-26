from dgu.graph import TextWorldGraph


def test_tw_graph_node():
    g = TextWorldGraph()
    node_id = g.add_node("game0", 0, "n1")
    assert node_id == 0
    assert g._graph.order() == 1
    assert g._graph.nodes[node_id]["game"] == "game0"
    assert g._graph.nodes[node_id]["walkthrough_step"] == 0
    assert g._graph.nodes[node_id]["label"] == "n1"
    assert g.get_node_labels() == ["n1"]

    node_id = g.add_node("game0", 1, "n2")
    assert node_id == 1
    assert g._graph.order() == 2
    assert g._graph.nodes[node_id]["game"] == "game0"
    assert g._graph.nodes[node_id]["walkthrough_step"] == 1
    assert g._graph.nodes[node_id]["label"] == "n2"
    assert g.get_node_labels() == ["n1", "n2"]

    node_id = g.add_node("game1", 0, "n1")
    assert node_id == 2
    assert g._graph.order() == 3
    assert g._graph.nodes[node_id]["game"] == "game1"
    assert g._graph.nodes[node_id]["walkthrough_step"] == 0
    assert g._graph.nodes[node_id]["label"] == "n1"
    assert g.get_node_labels() == ["n1", "n2", "n1"]

    g.remove_node(1)
    assert g._graph.order() == 2
    assert g.get_node_labels() == ["n1", "", "n1"]

    node_id = g.add_node("game2", 0, "n1")
    assert node_id == 1
    assert g._graph.order() == 3
    assert g._graph.nodes[node_id]["game"] == "game2"
    assert g._graph.nodes[node_id]["walkthrough_step"] == 0
    assert g._graph.nodes[node_id]["label"] == "n1"
    assert g.get_node_labels() == ["n1", "n1", "n1"]

    node_id = g.add_node("game2", 0, "n2")
    assert node_id == 3
    assert g._graph.order() == 4
    assert g._graph.nodes[node_id]["game"] == "game2"
    assert g._graph.nodes[node_id]["walkthrough_step"] == 0
    assert g._graph.nodes[node_id]["label"] == "n2"
    assert g.get_node_labels() == ["n1", "n1", "n1", "n2"]


def test_tw_graph_edge():
    g = TextWorldGraph()
    n1_id = g.add_node("game0", 0, "n1")
    n2_id = g.add_node("game0", 1, "n2")
    e1_id = g.add_edge(n1_id, n2_id, "game0", 1, "e1")
    assert e1_id == 0
    assert g._graph.number_of_edges() == 1
    assert g._graph[n1_id][n2_id]["id"] == e1_id
    assert g._graph[n1_id][n2_id]["game"] == "game0"
    assert g._graph[n1_id][n2_id]["walkthrough_step"] == 1
    assert g._graph[n1_id][n2_id]["label"] == "e1"

    n3_id = g.add_node("game0", 2, "n3")
    e2_id = g.add_edge(n2_id, n3_id, "game0", 2, "e2")
    assert e2_id == 1
    assert g._graph.number_of_edges() == 2
    assert g._graph[n2_id][n3_id]["id"] == e2_id
    assert g._graph[n2_id][n3_id]["game"] == "game0"
    assert g._graph[n2_id][n3_id]["walkthrough_step"] == 2
    assert g._graph[n2_id][n3_id]["label"] == "e2"

    n4_id = g.add_node("game1", 0, "n4")
    n5_id = g.add_node("game1", 1, "n5")
    e3_id = g.add_edge(n4_id, n5_id, "game1", 1, "e3")
    assert e3_id == 2
    assert g._graph.number_of_edges() == 3
    assert g._graph[n4_id][n5_id]["id"] == e3_id
    assert g._graph[n4_id][n5_id]["game"] == "game1"
    assert g._graph[n4_id][n5_id]["walkthrough_step"] == 1
    assert g._graph[n4_id][n5_id]["label"] == "e3"

    removed_e1_id = g.remove_edge(n1_id, n2_id)
    assert e1_id == removed_e1_id
    assert g._graph.number_of_edges() == 2

    n6_id = g.add_node("game1", 2, "n6")
    e4_id = g.add_edge(n5_id, n6_id, "game1", 2, "e4")
    assert e4_id == e1_id
    assert g._graph.number_of_edges() == 3
    assert g._graph[n5_id][n6_id]["id"] == e4_id
    assert g._graph[n5_id][n6_id]["game"] == "game1"
    assert g._graph[n5_id][n6_id]["walkthrough_step"] == 2
    assert g._graph[n5_id][n6_id]["label"] == "e4"


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
        {"type": "node-add", "node_id": 6, "timestamp": 6, "label": "cooked"},
        {
            "type": "edge-add",
            "edge_id": 5,
            "src_id": 4,
            "dst_id": 6,
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
        {"type": "node-add", "node_id": 13, "timestamp": 6, "label": "apple"},
        {
            "type": "edge-add",
            "edge_id": 10,
            "src_id": 13,
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
        "",
        "kitchen",
        "exit",
        "fridge",
        "closed",
        "living room",
    ]
    assert set(g.get_edge_labels()) == {
        (2, 1, "east of"),
        (3, 1, "in"),
        (3, 4, "is"),
        (1, 5, "east of"),
    }
    assert g.process_triplet_cmd("game0", 0, 2, "add , apple , kitchen , in") == [
        {"type": "node-add", "node_id": 0, "timestamp": 2, "label": "apple"},
        {
            "type": "edge-add",
            "edge_id": 0,
            "src_id": 0,
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
    ]
    assert set(g.get_edge_labels()) == {
        (0, 1, "in"),
        (2, 1, "east of"),
        (3, 1, "in"),
        (3, 4, "is"),
        (1, 5, "east of"),
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
    assert g.get_node_labels() == ["apple", "kitchen", "exit", "fridge", "closed", ""]
    assert set(g.get_edge_labels()) == {
        (0, 1, "in"),
        (2, 1, "east of"),
        (3, 1, "in"),
        (3, 4, "is"),
    }
    assert g.process_triplet_cmd(
        "game0", 0, 4, "add , kitchen , living room , east_of"
    ) == [
        {"type": "node-add", "node_id": 5, "timestamp": 4, "label": "living room"},
        {
            "type": "edge-add",
            "edge_id": 4,
            "src_id": 1,
            "dst_id": 5,
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
    ]
    assert set(g.get_edge_labels()) == {
        (0, 1, "in"),
        (2, 1, "east of"),
        (3, 1, "in"),
        (3, 4, "is"),
        (1, 5, "east of"),
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
        "",
        "living room",
    ]
    assert set(g.get_edge_labels()) == {
        (0, 1, "in"),
        (2, 1, "east of"),
        (3, 1, "in"),
        (1, 5, "east of"),
    }
    assert g.process_triplet_cmd("game0", 0, 6, "add , fridge , closed , is") == [
        {"type": "node-add", "node_id": 4, "timestamp": 6, "label": "closed"},
        {
            "type": "edge-add",
            "edge_id": 3,
            "src_id": 3,
            "dst_id": 4,
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
    ]
    assert set(g.get_edge_labels()) == {
        (0, 1, "in"),
        (2, 1, "east of"),
        (3, 1, "in"),
        (1, 5, "east of"),
        (3, 4, "is"),
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
        "",
        "fridge",
        "closed",
        "living room",
    ]
    assert set(g.get_edge_labels()) == {
        (0, 1, "in"),
        (3, 1, "in"),
        (1, 5, "east of"),
        (3, 4, "is"),
    }
    assert g.process_triplet_cmd("game0", 0, 8, "add , exit , kitchen , east_of") == [
        {"type": "node-add", "node_id": 2, "timestamp": 8, "label": "exit"},
        {
            "type": "edge-add",
            "edge_id": 1,
            "src_id": 2,
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
    ]
    assert set(g.get_edge_labels()) == {
        (0, 1, "in"),
        (2, 1, "east of"),
        (3, 1, "in"),
        (1, 5, "east of"),
        (3, 4, "is"),
    }


def test_tw_graph_get_subgraph():
    g = TextWorldGraph()
    g0_n1_id = g.add_node("game0", 0, "n1")
    g0_n2_id = g.add_node("game0", 0, "n2")
    g0_n3_id = g.add_node("game0", 0, "n3")
    g.add_edge(g0_n1_id, g0_n3_id, "game0", 0, "e1")
    g.add_node("game1", 0, "n1")
    g.remove_node(g0_n2_id)
    g2_n1_id = g.add_node("game2", 1, "n1")
    g2_n2_id = g.add_node("game2", 1, "n2")
    g.add_edge(g2_n1_id, g2_n2_id, "game2", 1, "e2")

    assert g.get_subgraph({("game0", 0)}) == ({0, 2}, {0})
    assert g.get_subgraph({("game1", 0)}) == ({3}, set())
    assert g.get_subgraph({("game2", 1)}) == ({1, 4}, {1})
