import pytest

from dgu.graph import Node, IsDstNode, ExitNode, process_triplet_cmd

from utils import EqualityDiGraph


@pytest.mark.parametrize(
    "graph,timestamp,cmd,expected_events",
    [
        # regular source and destination nodes
        (
            EqualityDiGraph(),
            0,
            "add , player , kitchen , in",
            [
                {
                    "type": "node-add",
                    "timestamp": 0,
                    "label": "player",
                    "before_graph": EqualityDiGraph(),
                    "after_graph": EqualityDiGraph({Node("player"): {}}),
                },
                {
                    "type": "node-add",
                    "timestamp": 0,
                    "label": "kitchen",
                    "before_graph": EqualityDiGraph({Node("player"): {}}),
                    "after_graph": EqualityDiGraph(
                        {Node("player"): {}, Node("kitchen"): {}}
                    ),
                },
                {
                    "type": "edge-add",
                    "src_id": 0,
                    "dst_id": 1,
                    "timestamp": 0,
                    "label": "in",
                    "before_graph": EqualityDiGraph(
                        {Node("player"): {}, Node("kitchen"): {}}
                    ),
                    "after_graph": EqualityDiGraph(
                        {
                            Node("player"): {
                                Node("kitchen"): {"label": "in", "last_update": 0}
                            }
                        }
                    ),
                },
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
                {
                    "type": "node-add",
                    "timestamp": 0,
                    "label": "exit",
                    "before_graph": EqualityDiGraph(),
                    "after_graph": EqualityDiGraph(
                        {ExitNode("exit", "west of", "kitchen"): {}}
                    ),
                },
                {
                    "type": "node-add",
                    "timestamp": 0,
                    "label": "kitchen",
                    "before_graph": EqualityDiGraph(
                        {ExitNode("exit", "west of", "kitchen"): {}}
                    ),
                    "after_graph": EqualityDiGraph(
                        {
                            ExitNode("exit", "west of", "kitchen"): {},
                            Node("kitchen"): {},
                        }
                    ),
                },
                {
                    "type": "edge-add",
                    "src_id": 0,
                    "dst_id": 1,
                    "timestamp": 0,
                    "label": "west of",
                    "before_graph": EqualityDiGraph(
                        {
                            ExitNode("exit", "west of", "kitchen"): {},
                            Node("kitchen"): {},
                        }
                    ),
                    "after_graph": EqualityDiGraph(
                        {
                            ExitNode("exit", "west of", "kitchen"): {
                                Node("kitchen"): {"label": "west of", "last_update": 0}
                            }
                        }
                    ),
                },
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
                {
                    "type": "node-add",
                    "timestamp": 1,
                    "label": "exit",
                    "before_graph": EqualityDiGraph(
                        {
                            ExitNode("exit", "east of", "kitchen"): {
                                Node("kitchen"): {"label": "east of", "last_update": 0}
                            }
                        }
                    ),
                    "after_graph": EqualityDiGraph(
                        {
                            ExitNode("exit", "east of", "kitchen"): {
                                Node("kitchen"): {"label": "east of", "last_update": 0}
                            },
                            ExitNode("exit", "west of", "kitchen"): {},
                        }
                    ),
                },
                {
                    "type": "edge-add",
                    "src_id": 2,
                    "dst_id": 1,
                    "timestamp": 1,
                    "label": "west of",
                    "before_graph": EqualityDiGraph(
                        {
                            ExitNode("exit", "east of", "kitchen"): {
                                Node("kitchen"): {"label": "east of", "last_update": 0}
                            },
                            ExitNode("exit", "west of", "kitchen"): {},
                        }
                    ),
                    "after_graph": EqualityDiGraph(
                        {
                            ExitNode("exit", "east of", "kitchen"): {
                                Node("kitchen"): {"label": "east of", "last_update": 0}
                            },
                            ExitNode("exit", "west of", "kitchen"): {
                                Node("kitchen"): {"label": "west of", "last_update": 1}
                            },
                        }
                    ),
                },
            ],
        ),
        # is destination node
        (
            EqualityDiGraph(),
            0,
            "add , steak , cooked , is",
            [
                {
                    "type": "node-add",
                    "timestamp": 0,
                    "label": "steak",
                    "before_graph": EqualityDiGraph(),
                    "after_graph": EqualityDiGraph({Node("steak"): {}}),
                },
                {
                    "type": "node-add",
                    "timestamp": 0,
                    "label": "cooked",
                    "before_graph": EqualityDiGraph({Node("steak"): {}}),
                    "after_graph": EqualityDiGraph(
                        {Node("steak"): {}, IsDstNode("cooked", "steak"): {}}
                    ),
                },
                {
                    "type": "edge-add",
                    "src_id": 0,
                    "dst_id": 1,
                    "timestamp": 0,
                    "label": "is",
                    "before_graph": EqualityDiGraph(
                        {Node("steak"): {}, IsDstNode("cooked", "steak"): {}}
                    ),
                    "after_graph": EqualityDiGraph(
                        {
                            Node("steak"): {
                                IsDstNode("cooked", "steak"): {
                                    "label": "is",
                                    "last_update": 0,
                                }
                            }
                        }
                    ),
                },
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
                {
                    "type": "node-add",
                    "timestamp": 2,
                    "label": "delicious",
                    "before_graph": EqualityDiGraph(
                        {
                            Node("steak"): {
                                IsDstNode("cooked", "steak"): {
                                    "label": "is",
                                    "last_update": 1,
                                }
                            }
                        }
                    ),
                    "after_graph": EqualityDiGraph(
                        {
                            Node("steak"): {
                                IsDstNode("cooked", "steak"): {
                                    "label": "is",
                                    "last_update": 1,
                                }
                            },
                            IsDstNode("delicious", "steak"): {},
                        }
                    ),
                },
                {
                    "type": "edge-add",
                    "src_id": 0,
                    "dst_id": 2,
                    "timestamp": 2,
                    "label": "is",
                    "before_graph": EqualityDiGraph(
                        {
                            Node("steak"): {
                                IsDstNode("cooked", "steak"): {
                                    "label": "is",
                                    "last_update": 1,
                                }
                            },
                            IsDstNode("delicious", "steak"): {},
                        }
                    ),
                    "after_graph": EqualityDiGraph(
                        {
                            Node("steak"): {
                                IsDstNode("cooked", "steak"): {
                                    "label": "is",
                                    "last_update": 1,
                                },
                                IsDstNode("delicious", "steak"): {
                                    "label": "is",
                                    "last_update": 2,
                                },
                            }
                        }
                    ),
                },
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
                    "timestamp": 1,
                    "label": "in",
                    "before_graph": EqualityDiGraph(
                        {
                            Node("player"): {
                                Node("kitchen"): {"label": "in", "last_update": 0}
                            }
                        }
                    ),
                    "after_graph": EqualityDiGraph(
                        {Node("player"): {}, Node("kitchen"): {}}
                    ),
                },
                {
                    "type": "node-delete",
                    "node_id": 1,
                    "timestamp": 1,
                    "label": "kitchen",
                    "before_graph": EqualityDiGraph(
                        {Node("player"): {}, Node("kitchen"): {}}
                    ),
                    "after_graph": EqualityDiGraph({Node("player"): {}}),
                },
                {
                    "type": "node-delete",
                    "node_id": 0,
                    "timestamp": 1,
                    "label": "player",
                    "before_graph": EqualityDiGraph({Node("player"): {}}),
                    "after_graph": EqualityDiGraph(),
                },
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
                    "timestamp": 3,
                    "label": "west of",
                    "before_graph": EqualityDiGraph(
                        {
                            ExitNode("exit", "east of", "kitchen"): {
                                Node("kitchen"): {"label": "east of", "last_update": 2}
                            },
                            ExitNode("exit", "west of", "kitchen"): {
                                Node("kitchen"): {"label": "west of", "last_update": 2}
                            },
                        }
                    ),
                    "after_graph": EqualityDiGraph(
                        {
                            ExitNode("exit", "east of", "kitchen"): {
                                Node("kitchen"): {"label": "east of", "last_update": 2}
                            },
                            ExitNode("exit", "west of", "kitchen"): {},
                        }
                    ),
                },
                {
                    "type": "node-delete",
                    "node_id": 1,
                    "timestamp": 3,
                    "label": "exit",
                    "before_graph": EqualityDiGraph(
                        {
                            ExitNode("exit", "east of", "kitchen"): {
                                Node("kitchen"): {"label": "east of", "last_update": 2}
                            },
                            ExitNode("exit", "west of", "kitchen"): {},
                        }
                    ),
                    "after_graph": EqualityDiGraph(
                        {
                            ExitNode("exit", "east of", "kitchen"): {
                                Node("kitchen"): {"label": "east of", "last_update": 2}
                            }
                        }
                    ),
                },
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
                {
                    "type": "edge-delete",
                    "src_id": 0,
                    "dst_id": 2,
                    "timestamp": 4,
                    "label": "is",
                    "before_graph": EqualityDiGraph(
                        {
                            Node("steak"): {
                                IsDstNode("cooked", "steak"): {
                                    "label": "is",
                                    "last_update": 3,
                                },
                                IsDstNode("delicious", "steak"): {
                                    "label": "is",
                                    "last_update": 3,
                                },
                            }
                        }
                    ),
                    "after_graph": EqualityDiGraph(
                        {
                            Node("steak"): {
                                IsDstNode("cooked", "steak"): {
                                    "label": "is",
                                    "last_update": 3,
                                }
                            },
                            IsDstNode("delicious", "steak"): {},
                        }
                    ),
                },
                {
                    "type": "node-delete",
                    "node_id": 2,
                    "timestamp": 4,
                    "label": "delicious",
                    "before_graph": EqualityDiGraph(
                        {
                            Node("steak"): {
                                IsDstNode("cooked", "steak"): {
                                    "label": "is",
                                    "last_update": 3,
                                }
                            },
                            IsDstNode("delicious", "steak"): {},
                        }
                    ),
                    "after_graph": EqualityDiGraph(
                        {
                            Node("steak"): {
                                IsDstNode("cooked", "steak"): {
                                    "label": "is",
                                    "last_update": 3,
                                }
                            }
                        }
                    ),
                },
            ],
        ),
    ],
)
def test_process_triplet_cmd(graph, timestamp, cmd, expected_events):
    assert process_triplet_cmd(graph, timestamp, cmd) == expected_events
