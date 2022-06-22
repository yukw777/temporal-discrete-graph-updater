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
        pretrained_word_embedding_path=f"{tmp_path}/test-fasttext.vec",
        word_vocab_path="tests/data/test_word_vocab.txt",
    )


@pytest.mark.parametrize(
    "batch,split_size,expected",
    [
        (
            TWCmdGenObsGenBatch(
                ((("g1", 0, 0),),),
                (TWCmdGenGraphEventStepInput(),),
                torch.tensor([[True]]),
            ),
            1,
            [
                TWCmdGenObsGenBatch(
                    ((("g1", 0, 0),),),
                    (TWCmdGenGraphEventStepInput(),),
                    torch.tensor([[True]]),
                )
            ],
        ),
        (
            TWCmdGenObsGenBatch(
                ((("g1", 0, 0),),),
                (TWCmdGenGraphEventStepInput(),),
                torch.tensor([[True]]),
            ),
            2,
            [
                TWCmdGenObsGenBatch(
                    ((("g1", 0, 0),),),
                    (TWCmdGenGraphEventStepInput(),),
                    torch.tensor([[True]]),
                )
            ],
        ),
        (
            TWCmdGenObsGenBatch(
                (
                    (("g1", 0, 0), ("g2", 0, 0), ("g3", 0, 0)),
                    (("g1", 1, 0), ("g2", 0, 1), ("g3", 1, 0)),
                    (("g1", 1, 1), ("g2", 0, 2), ("g3", 2, 0)),
                ),
                (
                    TWCmdGenGraphEventStepInput(),
                    TWCmdGenGraphEventStepInput(),
                    TWCmdGenGraphEventStepInput(),
                ),
                torch.tensor([[True] * 3, [True, False, True], [False, False, True]]),
            ),
            2,
            [
                TWCmdGenObsGenBatch(
                    (
                        (("g1", 0, 0), ("g2", 0, 0), ("g3", 0, 0)),
                        (("g1", 1, 0), ("g2", 0, 1), ("g3", 1, 0)),
                    ),
                    (TWCmdGenGraphEventStepInput(), TWCmdGenGraphEventStepInput()),
                    torch.tensor([[True] * 3, [True, False, True]]),
                ),
                TWCmdGenObsGenBatch(
                    ((("g1", 1, 1), ("g2", 0, 2), ("g3", 2, 0)),),
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
            [torch.tensor([EVENT_TYPE_ID_MAP["pad"]])],
            [torch.tensor([1])],
            [torch.tensor([0])],
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
            ["(pad, <none>, <none>, <none>)"],
        ),
        (
            [torch.tensor([EVENT_TYPE_ID_MAP["pad"]])],
            [torch.tensor([1])],
            [torch.tensor([0])],
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
            [torch.tensor([EVENT_TYPE_ID_MAP["start"]])],
            [torch.tensor([1])],
            [torch.tensor([0])],
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
            [torch.tensor([EVENT_TYPE_ID_MAP["start"]])],
            [torch.tensor([1])],
            [torch.tensor([0])],
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
            [torch.tensor([EVENT_TYPE_ID_MAP["end"]])],
            [torch.tensor([1])],
            [torch.tensor([0])],
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
            [torch.tensor([EVENT_TYPE_ID_MAP["end"]])],
            [torch.tensor([1])],
            [torch.tensor([0])],
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
            [torch.tensor([EVENT_TYPE_ID_MAP["node-add"]])],
            [torch.tensor([0])],
            [torch.tensor([0])],
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
            [torch.tensor([EVENT_TYPE_ID_MAP["node-add"]])],
            [torch.tensor([0])],
            [torch.tensor([0])],
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
            [torch.tensor([EVENT_TYPE_ID_MAP["node-delete"]])],
            [torch.tensor([0])],
            [torch.tensor([0])],
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
            [torch.tensor([EVENT_TYPE_ID_MAP["node-delete"]])],
            [torch.tensor([0])],
            [torch.tensor([0])],
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
            [torch.tensor([EVENT_TYPE_ID_MAP["edge-add"]])],
            [torch.tensor([1])],
            [torch.tensor([0])],
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
            [torch.tensor([EVENT_TYPE_ID_MAP["edge-add"]])],
            [torch.tensor([1])],
            [torch.tensor([0])],
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
            [torch.tensor([EVENT_TYPE_ID_MAP["edge-delete"]])],
            [torch.tensor([1])],
            [torch.tensor([0])],
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
            [torch.tensor([EVENT_TYPE_ID_MAP["edge-delete"]])],
            [torch.tensor([1])],
            [torch.tensor([0])],
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
                torch.tensor(
                    [EVENT_TYPE_ID_MAP["edge-add"], EVENT_TYPE_ID_MAP["edge-delete"]]
                ),
                torch.tensor(
                    [EVENT_TYPE_ID_MAP["edge-delete"], EVENT_TYPE_ID_MAP["edge-add"]]
                ),
            ],
            [torch.tensor([1, 0]), torch.tensor([0, 2])],
            [torch.tensor([0, 1]), torch.tensor([1, 3])],
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
                torch.tensor(
                    [EVENT_TYPE_ID_MAP["edge-add"], EVENT_TYPE_ID_MAP["edge-delete"]]
                ),
                torch.tensor(
                    [EVENT_TYPE_ID_MAP["edge-delete"], EVENT_TYPE_ID_MAP["edge-add"]]
                ),
            ],
            [torch.tensor([1, 0]), torch.tensor([0, 2])],
            [torch.tensor([0, 1]), torch.tensor([1, 3])],
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
                torch.tensor(
                    [EVENT_TYPE_ID_MAP["edge-add"], EVENT_TYPE_ID_MAP["edge-delete"]]
                ),
                torch.tensor(
                    [EVENT_TYPE_ID_MAP["edge-delete"], EVENT_TYPE_ID_MAP["edge-add"]]
                ),
            ],
            [torch.tensor([1, 0]), torch.tensor([0, 2])],
            [torch.tensor([0, 1]), torch.tensor([1, 3])],
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
                torch.tensor(
                    [EVENT_TYPE_ID_MAP["node-add"], EVENT_TYPE_ID_MAP["edge-delete"]]
                ),
                torch.tensor(
                    [EVENT_TYPE_ID_MAP["edge-add"], EVENT_TYPE_ID_MAP["node-delete"]]
                ),
            ],
            [torch.tensor([0, 0]), torch.tensor([0, 2])],
            [torch.tensor([0, 1]), torch.tensor([1, 0])],
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
                torch.tensor(
                    [EVENT_TYPE_ID_MAP["node-add"], EVENT_TYPE_ID_MAP["edge-delete"]]
                ),
                torch.tensor(
                    [EVENT_TYPE_ID_MAP["edge-add"], EVENT_TYPE_ID_MAP["node-delete"]]
                ),
            ],
            [torch.tensor([0, 0]), torch.tensor([0, 2])],
            [torch.tensor([0, 1]), torch.tensor([1, 0])],
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
