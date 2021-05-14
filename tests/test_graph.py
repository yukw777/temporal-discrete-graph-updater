import pytest

from dgu.graph import TextWorldGraph


def test_tw_graph_node():
    g = TextWorldGraph()

    # add
    g.add_node("player")
    assert g.get_node_id("player") == 0
    with pytest.raises(KeyError):
        g.get_node_id("kitchen")
    g.add_node("kitchen")
    assert g.get_node_id("kitchen") == 1
    assert g.order() == 2
    assert g.has_node("player")
    assert g.has_node("kitchen")
    assert g.nodes[0]["label"] == "player"
    assert not g.nodes[0]["removed"]
    assert g.nodes[1]["label"] == "kitchen"
    assert not g.nodes[1]["removed"]

    # remove
    g.remove_node("player")
    # we don't actually remove the node, we just mark it removed.
    assert g.order() == 2
    assert not g.has_node("player")
    assert g.has_node("kitchen")
    assert g.nodes[0]["label"] == "player"
    assert g.nodes[0]["removed"]
    assert g.nodes[1]["label"] == "kitchen"
    assert not g.nodes[1]["removed"]

    # remove the same node again, which is a noop
    g.remove_node("player")
    # we don't actually remove the node, we just mark it removed.
    assert g.order() == 2
    assert not g.has_node("player")
    assert g.has_node("kitchen")
    assert g.nodes[0]["label"] == "player"
    assert g.nodes[0]["removed"]
    assert g.nodes[1]["label"] == "kitchen"
    assert not g.nodes[1]["removed"]

    # add it back
    # this creates a new node
    g.add_node("player")
    assert g.get_node_id("player") == 2
    assert g.order() == 3
    assert g.has_node("player")
    assert g.has_node("kitchen")
    assert g.nodes[0]["label"] == "player"
    assert g.nodes[0]["removed"]
    assert g.nodes[1]["label"] == "kitchen"
    assert not g.nodes[1]["removed"]
    assert g.nodes[2]["label"] == "player"
    assert not g.nodes[2]["removed"]


def test_tw_graph_edge():
    g = TextWorldGraph()

    # add nodes and edge separately
    g.add_node("player")
    g.add_node("kitchen")
    g.add_edge("player", "in", "kitchen")
    assert g.order() == 2
    assert g.has_edge("player", "kitchen")
    assert not g.has_edge("kitchen", "player")

    # remove edge
    g.remove_edge("player", "kitchen")
    assert g.order() == 2
    assert not g.has_edge("player", "kitchen")
    assert not g.has_edge("kitchen", "player")

    # add edge back
    g.add_edge("player", "in", "kitchen")
    assert g.order() == 2
    assert g.has_edge("player", "kitchen")
    assert not g.has_edge("kitchen", "player")

    # add edge without adding nodes
    g.add_edge("apple", "on", "table")
    assert g.order() == 4
    assert g.has_edge("apple", "table")
    assert not g.has_edge("table", "apple")
