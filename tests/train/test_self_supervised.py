import pytest
import torch
import shutil
import torch.nn.functional as F

from torch_geometric.data import Batch

from tdgu.train.self_supervised import ObsGenSelfSupervisedTDGU
from tdgu.data import TWCmdGenGraphEventStepInput, TWCmdGenObsGenBatch
from tdgu.constants import EVENT_TYPE_ID_MAP


@pytest.fixture
def obs_gen_self_supervised_tdgu(tmp_path):
    shutil.copy2("tests/data/test-fasttext.vec", tmp_path)
    return ObsGenSelfSupervisedTDGU(
        text_encoder_conf={
            "pretrained_word_embedding_path": f"{tmp_path}/test-fasttext.vec",
            "word_vocab_path": "tests/data/test_word_vocab.txt",
        }
    )


@pytest.mark.parametrize(
    "batch,split_size,expected",
    [
        (
            TWCmdGenObsGenBatch(
                (("g1", 0),),
                (TWCmdGenGraphEventStepInput(),),
                torch.tensor([[True]]),
            ),
            1,
            [
                TWCmdGenObsGenBatch(
                    (("g1", 0),),
                    (TWCmdGenGraphEventStepInput(),),
                    torch.tensor([[True]]),
                )
            ],
        ),
        (
            TWCmdGenObsGenBatch(
                (("g1", 0),),
                (TWCmdGenGraphEventStepInput(),),
                torch.tensor([[True]]),
            ),
            2,
            [
                TWCmdGenObsGenBatch(
                    (("g1", 0),),
                    (TWCmdGenGraphEventStepInput(),),
                    torch.tensor([[True]]),
                )
            ],
        ),
        (
            TWCmdGenObsGenBatch(
                (("g1", 0), ("g2", 1), ("g1", 1)),
                (
                    TWCmdGenGraphEventStepInput(),
                    TWCmdGenGraphEventStepInput(),
                    TWCmdGenGraphEventStepInput(),
                    TWCmdGenGraphEventStepInput(),
                ),
                torch.tensor(
                    [
                        [True] * 3,
                        [True, False, True],
                        [False, False, True],
                        [False, False, True],
                    ]
                ),
            ),
            3,
            [
                TWCmdGenObsGenBatch(
                    (("g1", 0), ("g2", 1), ("g1", 1)),
                    (
                        TWCmdGenGraphEventStepInput(),
                        TWCmdGenGraphEventStepInput(),
                        TWCmdGenGraphEventStepInput(),
                    ),
                    torch.tensor(
                        [
                            [True] * 3,
                            [True, False, True],
                            [False, False, True],
                        ]
                    ),
                ),
                TWCmdGenObsGenBatch(
                    (("g1", 0), ("g2", 1), ("g1", 1)),
                    (TWCmdGenGraphEventStepInput(),),
                    torch.tensor([[False, False, True]]),
                ),
            ],
        ),
    ],
)
def test_obs_gen_tbptt_split_batch(
    obs_gen_self_supervised_tdgu, batch, split_size, expected
):
    assert obs_gen_self_supervised_tdgu.tbptt_split_batch(batch, split_size) == expected


@pytest.mark.parametrize(
    "event_type_id_list,src_id_list,dst_id_list,label_word_id_list,label_mask_list,"
    "batched_graph_list,batched_step_mask,expected",
    [
        (
            [
                F.one_hot(
                    torch.tensor([EVENT_TYPE_ID_MAP["pad"]]),
                    num_classes=len(EVENT_TYPE_ID_MAP),
                ).float()
            ],
            [F.one_hot(torch.tensor([1]), num_classes=2).float()],
            [F.one_hot(torch.tensor([0]), num_classes=2).float()],
            [F.one_hot(torch.tensor([[2, 7, 3]]), num_classes=17).float()],
            [torch.ones(1, 3).bool()],
            [
                Batch(
                    batch=torch.tensor([0, 0]),
                    x=F.one_hot(
                        torch.tensor([[2, 15, 3], [2, 13, 3]]), num_classes=17
                    ).float(),
                    node_label_mask=torch.ones(2, 3).bool(),
                    node_last_update=torch.tensor([1, 2]),
                    edge_index=torch.empty(2, 0).long(),
                    edge_attr=torch.empty(0, 0, 17),
                    edge_label_mask=torch.empty(0, 0).bool(),
                    edge_last_update=torch.empty(0).long(),
                )
            ],
            torch.tensor([True]),
            [""],
        ),
        (
            [
                F.one_hot(
                    torch.tensor([EVENT_TYPE_ID_MAP["pad"]]),
                    num_classes=len(EVENT_TYPE_ID_MAP),
                ).float()
            ],
            [F.one_hot(torch.tensor([1]), num_classes=2).float()],
            [F.one_hot(torch.tensor([0]), num_classes=2).float()],
            [F.one_hot(torch.tensor([[2, 7, 3]]), num_classes=17).float()],
            [torch.tensor([[True, True]])],
            [
                Batch(
                    batch=torch.tensor([0, 0]),
                    x=F.one_hot(
                        torch.tensor([[2, 15, 3], [2, 13, 3]]), num_classes=17
                    ).float(),
                    node_label_mask=torch.ones(2, 3).bool(),
                    node_last_update=torch.tensor([1, 2]),
                    edge_index=torch.empty(2, 0).long(),
                    edge_attr=torch.empty(0, 0, 17),
                    edge_label_mask=torch.empty(0, 0).bool(),
                    edge_last_update=torch.empty(0).long(),
                )
            ],
            torch.tensor([False]),
            [""],
        ),
        (
            [
                F.one_hot(
                    torch.tensor([EVENT_TYPE_ID_MAP["start"]]),
                    num_classes=len(EVENT_TYPE_ID_MAP),
                ).float()
            ],
            [F.one_hot(torch.tensor([1]), num_classes=2).float()],
            [F.one_hot(torch.tensor([0]), num_classes=2).float()],
            [F.one_hot(torch.tensor([[2, 7, 3]]), num_classes=17).float()],
            [torch.tensor([[True, True]])],
            [
                Batch(
                    batch=torch.tensor([0, 0]),
                    x=F.one_hot(
                        torch.tensor([[2, 15, 3], [2, 13, 3]]), num_classes=17
                    ).float(),
                    node_label_mask=torch.ones(2, 3).bool(),
                    node_last_update=torch.tensor([1, 2]),
                    edge_index=torch.empty(2, 0).long(),
                    edge_attr=torch.empty(0, 0, 17),
                    edge_label_mask=torch.empty(0, 0).bool(),
                    edge_last_update=torch.empty(0).long(),
                )
            ],
            torch.tensor([True]),
            ["(start, <none>, <none>, <none>)"],
        ),
        (
            [
                F.one_hot(
                    torch.tensor([EVENT_TYPE_ID_MAP["start"]]),
                    num_classes=len(EVENT_TYPE_ID_MAP),
                ).float()
            ],
            [F.one_hot(torch.tensor([1]), num_classes=2).float()],
            [F.one_hot(torch.tensor([0]), num_classes=2).float()],
            [F.one_hot(torch.tensor([[2, 7, 3]]), num_classes=17).float()],
            [torch.tensor([[True, True]])],
            [
                Batch(
                    batch=torch.tensor([0, 0]),
                    x=F.one_hot(
                        torch.tensor([[2, 15, 3], [2, 13, 3]]), num_classes=17
                    ).float(),
                    node_label_mask=torch.ones(2, 3).bool(),
                    node_last_update=torch.tensor([1, 2]),
                    edge_index=torch.empty(2, 0).long(),
                    edge_attr=torch.empty(0, 0, 17),
                    edge_label_mask=torch.empty(0, 0).bool(),
                    edge_last_update=torch.empty(0).long(),
                )
            ],
            torch.tensor([False]),
            [""],
        ),
        (
            [
                F.one_hot(
                    torch.tensor([EVENT_TYPE_ID_MAP["end"]]),
                    num_classes=len(EVENT_TYPE_ID_MAP),
                ).float()
            ],
            [F.one_hot(torch.tensor([1]), num_classes=2).float()],
            [F.one_hot(torch.tensor([0]), num_classes=2).float()],
            [F.one_hot(torch.tensor([[2, 7, 3]]), num_classes=17).float()],
            [torch.tensor([[True, True]])],
            [
                Batch(
                    batch=torch.tensor([0, 0]),
                    x=F.one_hot(
                        torch.tensor([[2, 15, 3], [2, 13, 3]]), num_classes=17
                    ).float(),
                    node_label_mask=torch.ones(2, 3).bool(),
                    node_last_update=torch.tensor([1, 2]),
                    edge_index=torch.empty(2, 0).long(),
                    edge_attr=torch.empty(0, 0, 17),
                    edge_label_mask=torch.empty(0, 0).bool(),
                    edge_last_update=torch.empty(0).long(),
                )
            ],
            torch.tensor([True]),
            ["(end, <none>, <none>, <none>)"],
        ),
        (
            [
                F.one_hot(
                    torch.tensor([EVENT_TYPE_ID_MAP["end"]]),
                    num_classes=len(EVENT_TYPE_ID_MAP),
                ).float()
            ],
            [F.one_hot(torch.tensor([1]), num_classes=2).float()],
            [F.one_hot(torch.tensor([0]), num_classes=2).float()],
            [F.one_hot(torch.tensor([[2, 7, 3]]), num_classes=17).float()],
            [torch.tensor([[True, True]])],
            [
                Batch(
                    batch=torch.tensor([0, 0]),
                    x=F.one_hot(
                        torch.tensor([[2, 15, 3], [2, 13, 3]]), num_classes=17
                    ).float(),
                    node_label_mask=torch.ones(2, 3).bool(),
                    node_last_update=torch.tensor([1, 2]),
                    edge_index=torch.empty(2, 0).long(),
                    edge_attr=torch.empty(0, 0, 17),
                    edge_label_mask=torch.empty(0, 0).bool(),
                    edge_last_update=torch.empty(0).long(),
                )
            ],
            torch.tensor([False]),
            [""],
        ),
        (
            [
                F.one_hot(
                    torch.tensor([EVENT_TYPE_ID_MAP["node-add"]]),
                    num_classes=len(EVENT_TYPE_ID_MAP),
                ).float()
            ],
            [torch.empty(1, 0)],
            [torch.empty(1, 0)],
            [F.one_hot(torch.tensor([[2, 8, 3]]), num_classes=17).float()],
            [torch.ones(1, 3).bool()],
            [
                Batch(
                    batch=torch.tensor([0]),
                    x=F.one_hot(torch.tensor([[2, 13, 3]]), num_classes=17).float(),
                    node_label_mask=torch.ones(1, 3).bool(),
                    node_last_update=torch.tensor([[1, 2]]),
                    edge_index=torch.empty(2, 0).long(),
                    edge_attr=torch.empty(0, 0, 17),
                    edge_label_mask=torch.empty(0, 0).bool(),
                    edge_last_update=torch.empty(0).long(),
                )
            ],
            torch.tensor([True]),
            ["(node-add, peter, <none>, <none>)"],
        ),
        (
            [
                F.one_hot(
                    torch.tensor([EVENT_TYPE_ID_MAP["node-add"]]),
                    num_classes=len(EVENT_TYPE_ID_MAP),
                ).float()
            ],
            [torch.empty(1, 0)],
            [torch.empty(1, 0)],
            [F.one_hot(torch.tensor([[2, 8, 3]]), num_classes=17).float()],
            [torch.ones(1, 3).bool()],
            [
                Batch(
                    batch=torch.tensor([0]),
                    x=F.one_hot(torch.tensor([[2, 13, 3]]), num_classes=17).float(),
                    node_label_mask=torch.ones(1, 3).bool(),
                    node_last_update=torch.tensor([[1, 2]]),
                    edge_index=torch.empty(2, 0).long(),
                    edge_attr=torch.empty(0, 0, 17),
                    edge_label_mask=torch.empty(0, 0).bool(),
                    edge_last_update=torch.empty(0).long(),
                )
            ],
            torch.tensor([False]),
            [""],
        ),
        (
            [
                F.one_hot(
                    torch.tensor([EVENT_TYPE_ID_MAP["node-delete"]]),
                    num_classes=len(EVENT_TYPE_ID_MAP),
                ).float()
            ],
            [F.one_hot(torch.tensor([0]), num_classes=1).float()],
            [torch.zeros(1, 1)],
            [F.one_hot(torch.tensor([[2, 8, 3]]), num_classes=17).float()],
            [torch.ones(1, 3).bool()],
            [
                Batch(
                    batch=torch.tensor([0]),
                    x=F.one_hot(torch.tensor([[2, 13, 3]]), num_classes=17).float(),
                    node_label_mask=torch.ones(1, 3).bool(),
                    node_last_update=torch.tensor([[1, 2]]),
                    edge_index=torch.empty(2, 0).long(),
                    edge_attr=torch.empty(0, 0, 17),
                    edge_label_mask=torch.empty(0, 0).bool(),
                    edge_last_update=torch.empty(0).long(),
                )
            ],
            torch.tensor([True]),
            ["(node-delete, player, <none>, <none>)"],
        ),
        (
            [
                F.one_hot(
                    torch.tensor([EVENT_TYPE_ID_MAP["node-delete"]]),
                    num_classes=len(EVENT_TYPE_ID_MAP),
                ).float()
            ],
            [F.one_hot(torch.tensor([0]), num_classes=1).float()],
            [torch.zeros(1, 1)],
            [F.one_hot(torch.tensor([[2, 8, 3]]), num_classes=17).float()],
            [torch.ones(1, 3).bool()],
            [
                Batch(
                    batch=torch.tensor([0]),
                    x=F.one_hot(torch.tensor([[2, 13, 3]]), num_classes=17).float(),
                    node_label_mask=torch.ones(1, 3).bool(),
                    node_last_update=torch.tensor([[1, 2]]),
                    edge_index=torch.empty(2, 0).long(),
                    edge_attr=torch.empty(0, 0, 17),
                    edge_label_mask=torch.empty(0, 0).bool(),
                    edge_last_update=torch.empty(0).long(),
                )
            ],
            torch.tensor([False]),
            [""],
        ),
        (
            [
                F.one_hot(
                    torch.tensor([EVENT_TYPE_ID_MAP["edge-add"]]),
                    num_classes=len(EVENT_TYPE_ID_MAP),
                ).float()
            ],
            [F.one_hot(torch.tensor([1]), num_classes=2).float()],
            [F.one_hot(torch.tensor([0]), num_classes=2).float()],
            [F.one_hot(torch.tensor([[2, 7, 3]]), num_classes=17).float()],
            [torch.tensor([[True, True]])],
            [
                Batch(
                    batch=torch.tensor([0, 0]),
                    x=F.one_hot(
                        torch.tensor([[2, 15, 3], [2, 13, 3]]), num_classes=17
                    ).float(),
                    node_label_mask=torch.ones(2, 3).bool(),
                    node_last_update=torch.tensor([1, 2]),
                    edge_index=torch.empty(2, 0).long(),
                    edge_attr=torch.empty(0, 0, 17),
                    edge_label_mask=torch.empty(0, 0).bool(),
                    edge_last_update=torch.empty(0).long(),
                )
            ],
            torch.tensor([True]),
            ["(edge-add, player, chopped, is)"],
        ),
        (
            [
                F.one_hot(
                    torch.tensor([EVENT_TYPE_ID_MAP["edge-add"]]),
                    num_classes=len(EVENT_TYPE_ID_MAP),
                ).float()
            ],
            [F.one_hot(torch.tensor([1]), num_classes=2).float()],
            [F.one_hot(torch.tensor([0]), num_classes=2).float()],
            [F.one_hot(torch.tensor([[2, 7, 3]]), num_classes=17).float()],
            [torch.tensor([[True, True]])],
            [
                Batch(
                    batch=torch.tensor([0, 0]),
                    x=F.one_hot(
                        torch.tensor([[2, 15, 3], [2, 13, 3]]), num_classes=17
                    ).float(),
                    node_label_mask=torch.ones(2, 3).bool(),
                    node_last_update=torch.tensor([1, 2]),
                    edge_index=torch.empty(2, 0).long(),
                    edge_attr=torch.empty(0, 0, 17),
                    edge_label_mask=torch.empty(0, 0).bool(),
                    edge_last_update=torch.empty(0).long(),
                )
            ],
            torch.tensor([False]),
            [""],
        ),
        (
            [
                F.one_hot(
                    torch.tensor([EVENT_TYPE_ID_MAP["edge-delete"]]),
                    num_classes=len(EVENT_TYPE_ID_MAP),
                ).float()
            ],
            [F.one_hot(torch.tensor([1]), num_classes=2).float()],
            [F.one_hot(torch.tensor([0]), num_classes=2).float()],
            [F.one_hot(torch.tensor([[2, 8, 3]]), num_classes=17).float()],
            [torch.ones(1, 3).bool()],
            [
                Batch(
                    batch=torch.tensor([0, 0]),
                    x=F.one_hot(
                        torch.tensor([[2, 15, 3], [2, 13, 3]]), num_classes=17
                    ).float(),
                    node_label_mask=torch.ones(2, 3).bool(),
                    node_last_update=torch.tensor([[0, 0], [0, 1]]),
                    edge_index=torch.tensor([[1], [0]]),
                    edge_attr=F.one_hot(
                        torch.tensor([[2, 7, 3]]), num_classes=17
                    ).float(),
                    edge_label_mask=torch.ones(1, 2).bool(),
                    edge_last_update=torch.tensor([[0, 2]]),
                )
            ],
            torch.tensor([True]),
            ["(edge-delete, player, chopped, is)"],
        ),
        (
            [
                F.one_hot(
                    torch.tensor([EVENT_TYPE_ID_MAP["edge-delete"]]),
                    num_classes=len(EVENT_TYPE_ID_MAP),
                ).float()
            ],
            [F.one_hot(torch.tensor([1]), num_classes=2).float()],
            [F.one_hot(torch.tensor([0]), num_classes=2).float()],
            [F.one_hot(torch.tensor([[2, 8, 3]]), num_classes=17).float()],
            [torch.ones(1, 3).bool()],
            [
                Batch(
                    batch=torch.tensor([0, 0]),
                    x=F.one_hot(
                        torch.tensor([[2, 15, 3], [2, 13, 3]]), num_classes=17
                    ).float(),
                    node_label_mask=torch.ones(2, 3).bool(),
                    node_last_update=torch.tensor([[0, 0], [0, 1]]),
                    edge_index=torch.tensor([[1], [0]]),
                    edge_attr=F.one_hot(
                        torch.tensor([[2, 7, 3]]), num_classes=17
                    ).float(),
                    edge_label_mask=torch.ones(1, 2).bool(),
                    edge_last_update=torch.tensor([[0, 2]]),
                )
            ],
            torch.tensor([False]),
            [""],
        ),
        (
            [
                F.one_hot(
                    torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["edge-add"],
                            EVENT_TYPE_ID_MAP["edge-delete"],
                        ]
                    ),
                    num_classes=len(EVENT_TYPE_ID_MAP),
                ).float(),
                F.one_hot(
                    torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["edge-delete"],
                            EVENT_TYPE_ID_MAP["edge-add"],
                        ]
                    ),
                    num_classes=len(EVENT_TYPE_ID_MAP),
                ).float(),
            ],
            [
                F.one_hot(torch.tensor([1, 0]), num_classes=4).float(),
                F.one_hot(torch.tensor([0, 2]), num_classes=4).float(),
            ],
            [
                F.one_hot(torch.tensor([0, 1]), num_classes=4).float(),
                F.one_hot(torch.tensor([1, 3]), num_classes=4).float(),
            ],
            [
                F.one_hot(torch.tensor([[2, 7, 3], [2, 3, 0]]), num_classes=17).float(),
                F.one_hot(
                    torch.tensor([[2, 3, 0], [2, 12, 3]]), num_classes=17
                ).float(),
            ],
            [
                torch.tensor([[True] * 3, [True, True, False]]),
                torch.tensor([[True, True, False], [True] * 3]),
            ],
            [
                Batch(
                    batch=torch.tensor([0, 0, 0, 0, 1, 1]),
                    x=F.one_hot(
                        torch.tensor(
                            [
                                [2, 15, 3],
                                [2, 13, 3],
                                [2, 13, 3],
                                [2, 8, 3],
                                [2, 13, 3],
                                [2, 14, 3],
                            ]
                        ),
                        num_classes=17,
                    ).float(),
                    node_label_mask=torch.ones(6, 3).bool(),
                    node_last_update=torch.tensor(
                        [[1, 0], [2, 1], [3, 1], [2, 3], [1, 2], [2, 3]]
                    ),
                    edge_index=torch.tensor([[4], [5]]),
                    edge_attr=F.one_hot(
                        torch.tensor([[2, 12, 3]]), num_classes=17
                    ).float(),
                    edge_label_mask=torch.ones(1, 3).bool(),
                    edge_last_update=torch.tensor([[2, 1]]),
                ),
                Batch(
                    batch=torch.tensor([0, 0, 1, 1, 1, 1]),
                    x=F.one_hot(
                        torch.tensor(
                            [
                                [2, 13, 3],
                                [2, 15, 3],
                                [2, 13, 3],
                                [2, 15, 3],
                                [2, 13, 3],
                                [2, 14, 3],
                            ]
                        ),
                        num_classes=17,
                    ).float(),
                    node_label_mask=torch.ones(6, 3).bool(),
                    node_last_update=torch.tensor(
                        [[1, 2], [1, 3], [2, 0], [2, 1], [2, 2], [2, 3]]
                    ),
                    edge_index=torch.tensor([[0], [1]]),
                    edge_attr=F.one_hot(
                        torch.tensor([[2, 7, 3]]), num_classes=17
                    ).float(),
                    edge_label_mask=torch.ones(1, 3).bool(),
                    edge_last_update=torch.tensor([[1, 1]]),
                ),
            ],
            torch.tensor([True, True]),
            [
                "(edge-add, player, chopped, is), (edge-delete, player, chopped, is)",
                "(edge-delete, player, inventory, in), "
                "(edge-add, player, inventory, in)",
            ],
        ),
        (
            [
                F.one_hot(
                    torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["edge-add"],
                            EVENT_TYPE_ID_MAP["edge-delete"],
                        ]
                    ),
                    num_classes=len(EVENT_TYPE_ID_MAP),
                ).float(),
                F.one_hot(
                    torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["edge-delete"],
                            EVENT_TYPE_ID_MAP["edge-add"],
                        ]
                    ),
                    num_classes=len(EVENT_TYPE_ID_MAP),
                ).float(),
            ],
            [
                F.one_hot(torch.tensor([1, 0]), num_classes=4).float(),
                F.one_hot(torch.tensor([0, 2]), num_classes=4).float(),
            ],
            [
                F.one_hot(torch.tensor([0, 1]), num_classes=4).float(),
                F.one_hot(torch.tensor([1, 3]), num_classes=4).float(),
            ],
            [
                F.one_hot(torch.tensor([[2, 7, 3], [2, 3, 0]]), num_classes=17).float(),
                F.one_hot(
                    torch.tensor([[2, 3, 0], [2, 12, 3]]), num_classes=17
                ).float(),
            ],
            [
                torch.tensor([[True] * 3, [True, True, False]]),
                torch.tensor([[True, True, False], [True] * 3]),
            ],
            [
                Batch(
                    batch=torch.tensor([0, 0, 0, 0, 1, 1]),
                    x=F.one_hot(
                        torch.tensor(
                            [
                                [2, 15, 3],
                                [2, 13, 3],
                                [2, 13, 3],
                                [2, 8, 3],
                                [2, 13, 3],
                                [2, 14, 3],
                            ]
                        ),
                        num_classes=17,
                    ).float(),
                    node_label_mask=torch.ones(6, 3).bool(),
                    node_last_update=torch.tensor(
                        [[1, 0], [2, 1], [3, 1], [2, 3], [1, 2], [2, 3]]
                    ),
                    edge_index=torch.tensor([[4], [5]]),
                    edge_attr=F.one_hot(
                        torch.tensor([[2, 12, 3]]), num_classes=17
                    ).float(),
                    edge_label_mask=torch.ones(1, 3).bool(),
                    edge_last_update=torch.tensor([[2, 1]]),
                ),
                Batch(
                    batch=torch.tensor([0, 0, 1, 1, 1, 1]),
                    x=F.one_hot(
                        torch.tensor(
                            [
                                [2, 13, 3],
                                [2, 15, 3],
                                [2, 13, 3],
                                [2, 15, 3],
                                [2, 13, 3],
                                [2, 14, 3],
                            ]
                        ),
                        num_classes=17,
                    ).float(),
                    node_label_mask=torch.ones(6, 3).bool(),
                    node_last_update=torch.tensor(
                        [[1, 2], [1, 3], [2, 0], [2, 1], [2, 2], [2, 3]]
                    ),
                    edge_index=torch.tensor([[0], [1]]),
                    edge_attr=F.one_hot(
                        torch.tensor([[2, 7, 3]]), num_classes=17
                    ).float(),
                    edge_label_mask=torch.ones(1, 3).bool(),
                    edge_last_update=torch.tensor([[1, 1]]),
                ),
            ],
            torch.tensor([True, True]),
            [
                "(edge-add, player, chopped, is), (edge-delete, player, chopped, is)",
                "(edge-delete, player, inventory, in), "
                "(edge-add, player, inventory, in)",
            ],
        ),
        (
            [
                F.one_hot(
                    torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["edge-add"],
                            EVENT_TYPE_ID_MAP["edge-delete"],
                        ]
                    ),
                    num_classes=len(EVENT_TYPE_ID_MAP),
                ).float(),
                F.one_hot(
                    torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["edge-delete"],
                            EVENT_TYPE_ID_MAP["edge-add"],
                        ]
                    ),
                    num_classes=len(EVENT_TYPE_ID_MAP),
                ).float(),
            ],
            [
                F.one_hot(torch.tensor([1, 0]), num_classes=4).float(),
                F.one_hot(torch.tensor([0, 2]), num_classes=4).float(),
            ],
            [
                F.one_hot(torch.tensor([0, 1]), num_classes=4).float(),
                F.one_hot(torch.tensor([1, 3]), num_classes=4).float(),
            ],
            [
                F.one_hot(torch.tensor([[2, 7, 3], [2, 3, 0]]), num_classes=17).float(),
                F.one_hot(
                    torch.tensor([[2, 3, 0], [2, 12, 3]]), num_classes=17
                ).float(),
            ],
            [
                torch.tensor([[True] * 3, [True, True, False]]),
                torch.tensor([[True, True, False], [True] * 3]),
            ],
            [
                Batch(
                    batch=torch.tensor([0, 0, 0, 0, 1, 1]),
                    x=F.one_hot(
                        torch.tensor(
                            [
                                [2, 15, 3],
                                [2, 13, 3],
                                [2, 13, 3],
                                [2, 8, 3],
                                [2, 13, 3],
                                [2, 14, 3],
                            ]
                        ),
                        num_classes=17,
                    ).float(),
                    node_label_mask=torch.ones(6, 3).bool(),
                    node_last_update=torch.tensor(
                        [[1, 0], [2, 1], [3, 1], [2, 3], [1, 2], [2, 3]]
                    ),
                    edge_index=torch.tensor([[4], [5]]),
                    edge_attr=F.one_hot(
                        torch.tensor([[2, 12, 3]]), num_classes=17
                    ).float(),
                    edge_label_mask=torch.ones(1, 3).bool(),
                    edge_last_update=torch.tensor([[2, 1]]),
                ),
                Batch(
                    batch=torch.tensor([0, 0, 1, 1, 1, 1]),
                    x=F.one_hot(
                        torch.tensor(
                            [
                                [2, 13, 3],
                                [2, 15, 3],
                                [2, 13, 3],
                                [2, 15, 3],
                                [2, 13, 3],
                                [2, 14, 3],
                            ]
                        ),
                        num_classes=17,
                    ).float(),
                    node_label_mask=torch.ones(6, 3).bool(),
                    node_last_update=torch.tensor(
                        [[1, 2], [1, 3], [2, 0], [2, 1], [2, 2], [2, 3]]
                    ),
                    edge_index=torch.tensor([[0], [1]]),
                    edge_attr=F.one_hot(
                        torch.tensor([[2, 7, 3]]), num_classes=17
                    ).float(),
                    edge_label_mask=torch.ones(1, 3).bool(),
                    edge_last_update=torch.tensor([[1, 1]]),
                ),
            ],
            torch.tensor([True, False]),
            [
                "(edge-add, player, chopped, is), (edge-delete, player, chopped, is)",
                "",
            ],
        ),
        (
            [
                F.one_hot(
                    torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["node-add"],
                            EVENT_TYPE_ID_MAP["edge-delete"],
                        ]
                    ),
                    num_classes=len(EVENT_TYPE_ID_MAP),
                ).float(),
                F.one_hot(
                    torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["edge-add"],
                            EVENT_TYPE_ID_MAP["node-delete"],
                        ]
                    ),
                    num_classes=len(EVENT_TYPE_ID_MAP),
                ).float(),
            ],
            [
                torch.cat(
                    [
                        torch.zeros(1, 4),
                        F.one_hot(torch.tensor([0]), num_classes=4).float(),
                    ],
                ),
                F.one_hot(torch.tensor([0, 2]), num_classes=4).float(),
            ],
            [
                torch.cat(
                    [
                        torch.zeros(1, 4),
                        F.one_hot(torch.tensor([1]), num_classes=4).float(),
                    ],
                ),
                torch.cat(
                    [
                        F.one_hot(torch.tensor([1]), num_classes=4).float(),
                        torch.zeros(1, 4),
                    ],
                ),
            ],
            [
                F.one_hot(torch.tensor([[2, 7, 3], [2, 3, 0]]), num_classes=17).float(),
                F.one_hot(torch.tensor([[2, 7, 3], [2, 3, 0]]), num_classes=17).float(),
            ],
            [
                torch.tensor([[True] * 3, [True, True, False]]),
                torch.tensor([[True] * 3, [True, True, False]]),
            ],
            [
                Batch(
                    batch=torch.tensor([0, 0, 0, 0, 1, 1]),
                    x=F.one_hot(
                        torch.tensor(
                            [
                                [2, 15, 3],
                                [2, 13, 3],
                                [2, 13, 3],
                                [2, 8, 3],
                                [2, 13, 3],
                                [2, 14, 3],
                            ]
                        ),
                        num_classes=17,
                    ).float(),
                    node_label_mask=torch.ones(6, 3).bool(),
                    node_last_update=torch.tensor(
                        [[0, 2], [0, 3], [0, 4], [0, 5], [1, 2], [1, 4]]
                    ),
                    edge_index=torch.tensor([[4], [5]]),
                    edge_attr=F.one_hot(
                        torch.tensor([[2, 12, 3]]), num_classes=17
                    ).float(),
                    edge_label_mask=torch.ones(1, 3).bool(),
                    edge_last_update=torch.tensor([[1, 5]]),
                ),
                Batch(
                    batch=torch.tensor([0, 0, 1, 1, 1, 1]),
                    x=F.one_hot(
                        torch.tensor(
                            [
                                [2, 13, 3],
                                [2, 15, 3],
                                [2, 13, 3],
                                [2, 15, 3],
                                [2, 13, 3],
                                [2, 14, 3],
                            ]
                        ),
                        num_classes=17,
                    ).float(),
                    node_label_mask=torch.ones(6, 3).bool(),
                    node_last_update=torch.tensor(
                        [[1, 3], [1, 4], [2, 1], [2, 2], [2, 3], [2, 4]]
                    ),
                    edge_index=torch.empty(2, 0).long(),
                    edge_attr=torch.empty(0, 0, 17),
                    edge_label_mask=torch.empty(0, 0).bool(),
                    edge_last_update=torch.empty(0, 2).long(),
                ),
            ],
            torch.ones(2, 2).bool(),
            [
                "(node-add, is, <none>, <none>), (edge-add, player, chopped, is)",
                "(edge-delete, player, inventory, in), "
                "(node-delete, player, <none>, <none>)",
            ],
        ),
        (
            [
                F.one_hot(
                    torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["node-add"],
                            EVENT_TYPE_ID_MAP["edge-delete"],
                        ]
                    ),
                    num_classes=len(EVENT_TYPE_ID_MAP),
                ).float(),
                F.one_hot(
                    torch.tensor(
                        [
                            EVENT_TYPE_ID_MAP["edge-add"],
                            EVENT_TYPE_ID_MAP["node-delete"],
                        ]
                    ),
                    num_classes=len(EVENT_TYPE_ID_MAP),
                ).float(),
            ],
            [
                torch.cat(
                    [
                        torch.zeros(1, 4),
                        F.one_hot(torch.tensor([0]), num_classes=4).float(),
                    ],
                ),
                F.one_hot(torch.tensor([0, 2]), num_classes=4).float(),
            ],
            [
                torch.cat(
                    [
                        torch.zeros(1, 4),
                        F.one_hot(torch.tensor([1]), num_classes=4).float(),
                    ],
                ),
                torch.cat(
                    [
                        F.one_hot(torch.tensor([1]), num_classes=4).float(),
                        torch.zeros(1, 4),
                    ],
                ),
            ],
            [
                F.one_hot(torch.tensor([[2, 7, 3], [2, 3, 0]]), num_classes=17).float(),
                F.one_hot(torch.tensor([[2, 7, 3], [2, 3, 0]]), num_classes=17).float(),
            ],
            [
                torch.tensor([[True] * 3, [True, True, False]]),
                torch.tensor([[True] * 3, [True, True, False]]),
            ],
            [
                Batch(
                    batch=torch.tensor([0, 0, 0, 0, 1, 1]),
                    x=F.one_hot(
                        torch.tensor(
                            [
                                [2, 15, 3],
                                [2, 13, 3],
                                [2, 13, 3],
                                [2, 8, 3],
                                [2, 13, 3],
                                [2, 14, 3],
                            ]
                        ),
                        num_classes=17,
                    ).float(),
                    node_label_mask=torch.ones(6, 3).bool(),
                    node_last_update=torch.tensor(
                        [[0, 2], [0, 3], [0, 4], [0, 5], [1, 2], [1, 4]]
                    ),
                    edge_index=torch.tensor([[4], [5]]),
                    edge_attr=F.one_hot(
                        torch.tensor([[2, 12, 3]]), num_classes=17
                    ).float(),
                    edge_label_mask=torch.ones(1, 3).bool(),
                    edge_last_update=torch.tensor([[1, 5]]),
                ),
                Batch(
                    batch=torch.tensor([0, 0, 1, 1, 1, 1]),
                    x=F.one_hot(
                        torch.tensor(
                            [
                                [2, 13, 3],
                                [2, 15, 3],
                                [2, 13, 3],
                                [2, 15, 3],
                                [2, 13, 3],
                                [2, 14, 3],
                            ]
                        ),
                        num_classes=17,
                    ).float(),
                    node_label_mask=torch.ones(6, 3).bool(),
                    node_last_update=torch.tensor(
                        [[1, 3], [1, 4], [2, 1], [2, 2], [2, 3], [2, 4]]
                    ),
                    edge_index=torch.empty(2, 0).long(),
                    edge_attr=torch.empty(0, 0, 17),
                    edge_label_mask=torch.empty(0, 0).bool(),
                    edge_last_update=torch.empty(0, 2).long(),
                ),
            ],
            torch.tensor([False, True]),
            [
                "",
                "(edge-delete, player, inventory, in), "
                "(node-delete, player, <none>, <none>)",
            ],
        ),
    ],
)
def test_obs_gen_decode_graph_events(
    obs_gen_self_supervised_tdgu,
    event_type_id_list,
    src_id_list,
    dst_id_list,
    label_word_id_list,
    label_mask_list,
    batched_graph_list,
    batched_step_mask,
    expected,
):
    assert (
        obs_gen_self_supervised_tdgu.decode_graph_events(
            event_type_id_list,
            src_id_list,
            dst_id_list,
            label_word_id_list,
            label_mask_list,
            batched_graph_list,
            batched_step_mask,
        )
        == expected
    )
