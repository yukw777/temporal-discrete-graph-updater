import pytest
import torch
import shutil

from torch_geometric.data.batch import Batch

from dgu.nn.graph_updater import StaticLabelDiscreteGraphUpdater
from dgu.constants import EVENT_TYPES, EVENT_TYPE_ID_MAP
from dgu.data import TWCmdGenTemporalStepInput


class MockUUID:
    def __init__(self):
        self.id = 0

    def __call__(self):
        uuid = f"added-node-{self.id}"
        self.id += 1
        return uuid


@pytest.fixture
def sldgu(tmp_path):
    shutil.copy2("tests/data/test-fasttext.vec", tmp_path)
    return StaticLabelDiscreteGraphUpdater(
        pretrained_word_embedding_path=f"{tmp_path}/test-fasttext.vec",
        word_vocab_path="tests/data/test_word_vocab.txt",
        node_vocab_path="tests/data/test_node_vocab.txt",
        relation_vocab_path="tests/data/test_relation_vocab.txt",
    )


@pytest.mark.parametrize("batch,seq_len", [(1, 10), (8, 24)])
def test_sldgu_encode_text(sldgu, batch, seq_len):
    assert sldgu.encode_text(
        torch.randint(13, (batch, seq_len)), torch.randint(2, (batch, seq_len)).float()
    ).size() == (batch, seq_len, sldgu.hparams.hidden_dim)


@pytest.mark.parametrize(
    "batch,num_node,obs_len,prev_action_len",
    [(1, 0, 10, 5), (8, 0, 20, 10), (8, 1, 20, 10), (8, 10, 20, 10)],
)
def test_sldgu_f_delta(sldgu, batch, num_node, obs_len, prev_action_len):
    delta_g = sldgu.f_delta(
        torch.rand(batch, num_node, sldgu.hparams.hidden_dim),
        # the first node is always the pad node, so make sure to mask that to test
        torch.cat(
            [torch.zeros(batch, 1), torch.randint(2, (batch, num_node - 1)).float()],
            dim=-1,
        )
        if num_node != 0
        else torch.randint(2, (batch, num_node)).float(),
        torch.rand(batch, obs_len, sldgu.hparams.hidden_dim),
        torch.randint(2, (batch, obs_len)).float(),
        torch.rand(batch, prev_action_len, sldgu.hparams.hidden_dim),
        torch.randint(2, (batch, prev_action_len)).float(),
    )
    assert delta_g.size() == (batch, 4 * sldgu.hparams.hidden_dim)
    assert delta_g.isnan().sum() == 0


@pytest.mark.parametrize(
    "batch_size,obs_len,prev_action_len,event_type_ids,event_src_ids,event_dst_ids,"
    "batch,prev_num_node,num_node,num_edge",
    [
        (
            1,
            10,
            5,
            torch.tensor([EVENT_TYPE_ID_MAP["start"]]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.empty(0).long(),
            0,
            0,
            0,
        ),
        (
            4,
            10,
            5,
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["start"],
                ]
            ),
            torch.tensor([0, 0, 0, 0]),
            torch.tensor([0, 0, 0, 0]),
            torch.empty(0).long(),
            0,
            0,
            0,
        ),
        (
            1,
            10,
            5,
            torch.tensor([EVENT_TYPE_ID_MAP["start"]]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([0, 0]),
            0,
            2,
            0,
        ),
        (
            1,
            10,
            5,
            torch.tensor([EVENT_TYPE_ID_MAP["start"]]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([0, 0]),
            0,
            2,
            1,
        ),
        (
            1,
            10,
            5,
            torch.tensor([EVENT_TYPE_ID_MAP["node-add"]]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([0] * 9),
            7,
            9,
            1,
        ),
        (
            1,
            10,
            5,
            torch.tensor([EVENT_TYPE_ID_MAP["node-delete"]]),
            torch.tensor([4]),
            torch.tensor([0]),
            torch.tensor([0] * 9),
            7,
            9,
            1,
        ),
        (
            1,
            10,
            5,
            torch.tensor([EVENT_TYPE_ID_MAP["edge-add"]]),
            torch.tensor([2]),
            torch.tensor([6]),
            torch.tensor([0] * 9),
            7,
            9,
            4,
        ),
        (
            1,
            10,
            5,
            torch.tensor([EVENT_TYPE_ID_MAP["edge-delete"]]),
            torch.tensor([2]),
            torch.tensor([6]),
            torch.tensor([0] * 9),
            7,
            9,
            4,
        ),
        (
            4,
            12,
            8,
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["end"],
                ]
            ),
            torch.tensor([0, 0, 3, 0]),
            torch.tensor([0, 0, 5, 0]),
            torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3]),
            6,
            12,
            10,
        ),
        (
            4,
            12,
            8,
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["edge-delete"],
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["node-delete"],
                ]
            ),
            torch.tensor([2, 0, 3, 4]),
            torch.tensor([4, 0, 5, 0]),
            torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3]),
            6,
            12,
            10,
        ),
    ],
)
@pytest.mark.parametrize("encoded_textual_input", [True, False])
@pytest.mark.parametrize("decoder_hidden", [True, False])
def test_sldgu_forward(
    sldgu,
    decoder_hidden,
    encoded_textual_input,
    batch_size,
    obs_len,
    prev_action_len,
    event_type_ids,
    event_src_ids,
    event_dst_ids,
    batch,
    prev_num_node,
    num_node,
    num_edge,
):
    num_label = len(sldgu.labels)
    encoded_obs = (
        torch.rand(batch_size, obs_len, sldgu.hparams.hidden_dim)
        if encoded_textual_input
        else None
    )
    encoded_prev_action = (
        torch.rand(batch_size, prev_action_len, sldgu.hparams.hidden_dim)
        if encoded_textual_input
        else None
    )
    if num_edge > 0:
        edge_index = torch.randint(num_node, (2, num_edge)).unique(dim=1)
        actual_num_edge = edge_index.size(1)
    else:
        edge_index = torch.empty(2, 0).long()
        actual_num_edge = num_edge
    results = sldgu(
        torch.randint(2, (batch_size, obs_len)).float(),
        torch.randint(2, (batch_size, prev_action_len)).float(),
        torch.randint(10, (batch_size,)).float(),
        event_type_ids,
        event_src_ids,
        event_dst_ids,
        torch.randint(len(sldgu.labels), (batch_size,)),
        torch.rand(prev_num_node, sldgu.tgn.memory_dim),
        torch.rand(num_node, sldgu.tgn.event_embedding_dim)
        if num_node > 0
        else torch.empty(num_node, sldgu.tgn.event_embedding_dim),
        torch.randint(num_node, (prev_num_node,))
        if num_node > 0
        else torch.empty(prev_num_node).long(),
        torch.randint(2, (prev_num_node,)),
        edge_index,
        torch.rand(actual_num_edge, sldgu.tgn.event_embedding_dim),
        torch.randint(10, (actual_num_edge,)),
        batch,
        decoder_hidden=torch.rand(batch_size, sldgu.hparams.hidden_dim)
        if decoder_hidden
        else None,
        obs_word_ids=None
        if encoded_textual_input
        else torch.randint(
            len(sldgu.preprocessor.word_to_id_dict), (batch_size, obs_len)
        ),
        prev_action_word_ids=None
        if encoded_textual_input
        else torch.randint(
            len(sldgu.preprocessor.word_to_id_dict), (batch_size, prev_action_len)
        ),
        encoded_obs=encoded_obs,
        encoded_prev_action=encoded_prev_action,
    )
    assert results["event_type_logits"].size() == (batch_size, len(EVENT_TYPES))
    if batch.numel() > 0:
        max_sub_graph_num_node = batch.bincount().max().item()
    else:
        max_sub_graph_num_node = 0
    assert results["src_logits"].size() == (batch_size, max_sub_graph_num_node)
    assert results["dst_logits"].size() == (batch_size, max_sub_graph_num_node)
    assert results["label_logits"].size() == (batch_size, num_label)
    assert results["new_decoder_hidden"].size() == (
        batch_size,
        sldgu.hparams.hidden_dim,
    )
    if encoded_textual_input:
        assert results["encoded_obs"].equal(encoded_obs)
        assert results["encoded_prev_action"].equal(encoded_prev_action)
    else:
        assert results["encoded_obs"].size() == (
            batch_size,
            obs_len,
            sldgu.hparams.hidden_dim,
        )
        assert results["encoded_prev_action"].size() == (
            batch_size,
            prev_action_len,
            sldgu.hparams.hidden_dim,
        )


@pytest.mark.parametrize(
    "node_embeddings,batch,batch_size,expected_batch_node_embeddings,"
    "expected_batch_mask",
    [
        (
            torch.tensor(
                [[2] * 4, [3] * 4, [4] * 4, [5] * 4, [6] * 4, [7] * 4]
            ).float(),
            torch.tensor([0, 1, 1, 2, 2, 2]),
            3,
            torch.tensor(
                [
                    [[2] * 4, [0] * 4, [0] * 4],
                    [[3] * 4, [4] * 4, [0] * 4],
                    [[5] * 4, [6] * 4, [7] * 4],
                ]
            ).float(),
            torch.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]]).float(),
        ),
        (
            torch.tensor(
                [[2] * 4, [3] * 4, [4] * 4, [5] * 4, [6] * 4, [7] * 4]
            ).float(),
            torch.tensor([0, 0, 0, 1, 2, 2]),
            3,
            torch.tensor(
                [
                    [[2] * 4, [3] * 4, [4] * 4],
                    [[5] * 4, [0] * 4, [0] * 4],
                    [[6] * 4, [7] * 4, [0] * 4],
                ]
            ).float(),
            torch.tensor([[1, 1, 1], [1, 0, 0], [1, 1, 0]]).float(),
        ),
        (
            torch.tensor(
                [[2] * 4, [3] * 4, [4] * 4, [5] * 4, [6] * 4, [7] * 4]
            ).float(),
            torch.tensor([0, 0, 1, 1, 1, 1]),
            4,
            torch.tensor(
                [
                    [[2] * 4, [3] * 4, [0] * 4, [0] * 4],
                    [[4] * 4, [5] * 4, [6] * 4, [7] * 4],
                    [[0] * 4, [0] * 4, [0] * 4, [0] * 4],
                    [[0] * 4, [0] * 4, [0] * 4, [0] * 4],
                ]
            ).float(),
            torch.tensor(
                [[1, 1, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]
            ).float(),
        ),
        (
            torch.empty(0, 4),
            torch.empty(0).long(),
            4,
            torch.zeros(4, 0, 4),
            torch.zeros(4, 0),
        ),
    ],
)
def test_sldgu_batchify_node_embeddings(
    node_embeddings,
    batch,
    batch_size,
    expected_batch_node_embeddings,
    expected_batch_mask,
):
    (
        batch_node_embeddings,
        batch_mask,
    ) = StaticLabelDiscreteGraphUpdater.batchify_node_embeddings(
        node_embeddings, batch, batch_size
    )
    assert batch_node_embeddings.equal(expected_batch_node_embeddings)
    assert batch_mask.equal(expected_batch_mask)


@pytest.mark.parametrize("batch,num_node", [(1, 5), (8, 12)])
def test_sldgu_calculate_loss(sldgu, batch, num_node):
    assert (
        sldgu.calculate_loss(
            torch.rand(batch, len(EVENT_TYPES)),
            torch.randint(len(EVENT_TYPES), (batch,)),
            torch.rand(batch, num_node),
            torch.randint(num_node, (batch,)),
            torch.rand(batch, num_node),
            torch.randint(num_node, (batch,)),
            torch.rand(batch, len(sldgu.labels)),
            torch.randint(len(sldgu.labels), (batch,)),
            torch.randint(2, (batch,)),
            torch.randint(2, (batch,)),
            torch.randint(2, (batch,)),
        ).size()
        == tuple()
    )


@pytest.mark.parametrize("batch,num_node", [(1, 5), (8, 12)])
def test_sldgu_calculate_f1s(sldgu, batch, num_node):
    results = sldgu.calculate_f1s(
        torch.rand(batch, len(EVENT_TYPES)),
        torch.randint(len(EVENT_TYPES), (batch,)),
        torch.rand(batch, num_node),
        torch.randint(num_node, (batch,)),
        torch.rand(batch, num_node),
        torch.randint(num_node, (batch,)),
        torch.rand(batch, len(sldgu.labels)),
        torch.randint(len(sldgu.labels), (batch,)),
        torch.randint(2, (batch,)),
        torch.randint(2, (batch,)),
    )
    assert results["event_type_f1"].size() == tuple()
    if "src_node_f1" in results:
        assert results["src_node_f1"].size() == tuple()
    if "dst_node_f1" in results:
        assert results["dst_node_f1"].size() == tuple()
    assert results["label_f1"].size() == tuple()


@pytest.mark.parametrize(
    "event_type_ids,src_ids,dst_ids,label_ids,batched_graph,expected",
    [
        (
            torch.tensor([EVENT_TYPE_ID_MAP["node-add"]]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([1]),  # player
            Batch(
                batch=torch.tensor([0]),
                x=torch.tensor([[9] * 300]).float(),
                edge_index=torch.empty(2, 0).long(),
                edge_attr=torch.empty(0, 300),
                edge_last_update=torch.empty(0),
            ),
            ([], []),
        ),
        (
            torch.tensor([EVENT_TYPE_ID_MAP["node-delete"]]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([1]),  # player
            Batch(
                batch=torch.tensor([0]),
                x=torch.tensor([[9] * 300]).float(),
                edge_index=torch.empty(2, 0).long(),
                edge_attr=torch.empty(0, 300),
                edge_last_update=torch.empty(0),
            ),
            ([], []),
        ),
        (
            torch.tensor([EVENT_TYPE_ID_MAP["edge-add"]]),
            torch.tensor([1]),
            torch.tensor([0]),
            torch.tensor([5]),  # is
            Batch(
                batch=torch.tensor([0, 0]),
                x=torch.tensor([[11] * 300, [9] * 300]).float(),
                edge_index=torch.empty(2, 0).long(),
                edge_attr=torch.empty(0, 300),
                edge_last_update=torch.empty(0),
            ),
            (["add , player , chopped , is"], [["add", "player", "chopped", "is"]]),
        ),
        (
            torch.tensor([EVENT_TYPE_ID_MAP["edge-delete"]]),
            torch.tensor([1]),
            torch.tensor([0]),
            torch.tensor([5]),  # is
            Batch(
                batch=torch.tensor([0, 0]),
                x=torch.tensor([[11] * 300, [9] * 300]).float(),
                edge_index=torch.empty(2, 0).long(),
                edge_attr=torch.empty(0, 300),
                edge_last_update=torch.empty(0),
            ),
            (
                ["delete , player , chopped , is"],
                [["delete", "player", "chopped", "is"]],
            ),
        ),
        (
            torch.tensor(
                [EVENT_TYPE_ID_MAP["edge-add"], EVENT_TYPE_ID_MAP["edge-delete"]]
            ),
            torch.tensor([1, 0]),
            torch.tensor([0, 1]),
            torch.tensor([5, 4]),  # [is, in]
            Batch(
                batch=torch.tensor([0, 0, 0, 0, 1, 1]),
                x=torch.tensor(
                    [
                        [11] * 300,
                        [9] * 300,
                        [9] * 300,
                        [10] * 300,
                        [9] * 300,
                        [10] * 300,
                    ]
                ).float(),
                edge_index=torch.empty(2, 0).long(),
                edge_attr=torch.empty(0, 300),
                edge_last_update=torch.empty(0),
            ),
            (
                ["add , player , chopped , is", "delete , player , inventory , in"],
                [
                    ["add", "player", "chopped", "is"],
                    ["delete", "player", "inventory", "in"],
                ],
            ),
        ),
    ],
)
def test_sldgu_generate_graph_triples(
    sldgu,
    event_type_ids,
    src_ids,
    dst_ids,
    label_ids,
    batched_graph,
    expected,
):
    assert (
        sldgu.generate_graph_triples(
            event_type_ids, src_ids, dst_ids, label_ids, batched_graph
        )
        == expected
    )


@pytest.mark.parametrize(
    "event_type_id_seq,src_id_seq,dst_id_seq,label_id_seq,batched_graph_seq,expected",
    [
        (
            [torch.tensor([EVENT_TYPE_ID_MAP["node-add"]])],
            [torch.tensor([0])],
            [torch.tensor([0])],
            [torch.tensor([0])],
            [
                Batch(
                    batch=torch.tensor([0]),
                    x=torch.tensor([[9] * 300]).float(),
                    edge_index=torch.empty(2, 0).long(),
                    edge_attr=torch.empty(0, 300),
                    edge_last_update=torch.empty(0),
                )
            ],
            ([[]], [[]]),
        ),
        (
            [torch.tensor([EVENT_TYPE_ID_MAP["node-delete"]])],
            [torch.tensor([0])],
            [torch.tensor([0])],
            [torch.tensor([0])],
            [
                Batch(
                    batch=torch.tensor([0]),
                    x=torch.tensor([[9] * 300]).float(),
                    edge_index=torch.empty(2, 0).long(),
                    edge_attr=torch.empty(0, 300),
                    edge_last_update=torch.empty(0),
                )
            ],
            ([[]], [[]]),
        ),
        (
            [torch.tensor([EVENT_TYPE_ID_MAP["edge-add"]])],
            [torch.tensor([1])],
            [torch.tensor([0])],
            [torch.tensor([5])],
            [
                Batch(
                    batch=torch.tensor([0, 0]),
                    x=torch.tensor([[11] * 300, [9] * 300]).float(),
                    edge_index=torch.empty(2, 0).long(),
                    edge_attr=torch.empty(0, 300),
                    edge_last_update=torch.empty(0),
                )
            ],
            (
                [["add , player , chopped , is"]],
                [["add", "player", "chopped", "is"]],
            ),
        ),
        (
            [torch.tensor([EVENT_TYPE_ID_MAP["edge-delete"]])],
            [torch.tensor([1])],
            [torch.tensor([0])],
            [torch.tensor([5])],
            [
                Batch(
                    batch=torch.tensor([0, 0]),
                    x=torch.tensor([[11] * 300, [9] * 300]).float(),
                    edge_index=torch.empty(2, 0).long(),
                    edge_attr=torch.empty(0, 300),
                    edge_last_update=torch.empty(0),
                )
            ],
            (
                [["delete , player , chopped , is"]],
                [["delete", "player", "chopped", "is"]],
            ),
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
            [torch.tensor([5, 4]), torch.tensor([5, 4])],
            [
                Batch(
                    batch=torch.tensor([0, 0, 0, 0, 1, 1]),
                    x=torch.tensor(
                        [
                            [11] * 300,
                            [9] * 300,
                            [9] * 300,
                            [10] * 300,
                            [9] * 300,
                            [10] * 300,
                        ]
                    ).float(),
                    edge_index=torch.empty(2, 0).long(),
                    edge_attr=torch.empty(0, 300),
                    edge_last_update=torch.empty(0),
                ),
                Batch(
                    batch=torch.tensor([0, 0, 1, 1, 1, 1]),
                    x=torch.tensor(
                        [
                            [9] * 300,
                            [11] * 300,
                            [9] * 300,
                            [10] * 300,
                            [9] * 300,
                            [10] * 300,
                        ]
                    ).float(),
                    edge_index=torch.empty(2, 0).long(),
                    edge_attr=torch.empty(0, 300),
                    edge_last_update=torch.empty(0),
                ),
            ],
            (
                [
                    ["add , player , chopped , is", "delete , player , chopped , is"],
                    [
                        "delete , player , inventory , in",
                        "add , player , inventory , in",
                    ],
                ],
                [
                    [
                        "add",
                        "player",
                        "chopped",
                        "is",
                        "<sep>",
                        "delete",
                        "player",
                        "chopped",
                        "is",
                    ],
                    [
                        "delete",
                        "player",
                        "inventory",
                        "in",
                        "<sep>",
                        "add",
                        "player",
                        "inventory",
                        "in",
                    ],
                ],
            ),
        ),
    ],
)
def test_sldgu_generate_batch_graph_triples_seq(
    sldgu,
    event_type_id_seq,
    src_id_seq,
    dst_id_seq,
    label_id_seq,
    batched_graph_seq,
    expected,
):
    assert (
        sldgu.generate_batch_graph_triples_seq(
            event_type_id_seq, src_id_seq, dst_id_seq, label_id_seq, batched_graph_seq
        )
        == expected
    )


@pytest.mark.parametrize(
    "groundtruth_cmd_seq,expected",
    [
        (
            (("add , player , kitchen , in",),),
            [["add", "player", "kitchen", "in"]],
        ),
        (
            (("add , player , kitchen , in", "delete , player , kitchen , in"),),
            [
                [
                    "add",
                    "player",
                    "kitchen",
                    "in",
                    "<sep>",
                    "delete",
                    "player",
                    "kitchen",
                    "in",
                ]
            ],
        ),
        (
            (
                ("add , player , kitchen , in", "delete , player , kitchen , in"),
                (
                    "add , player , kitchen , in",
                    "add , table , kitchen , in",
                    "delete , player , kitchen , in",
                ),
            ),
            [
                [
                    "add",
                    "player",
                    "kitchen",
                    "in",
                    "<sep>",
                    "delete",
                    "player",
                    "kitchen",
                    "in",
                ],
                [
                    "add",
                    "player",
                    "kitchen",
                    "in",
                    "<sep>",
                    "add",
                    "table",
                    "kitchen",
                    "in",
                    "<sep>",
                    "delete",
                    "player",
                    "kitchen",
                    "in",
                ],
            ],
        ),
    ],
)
def test_sldgu_generate_batch_groundtruth_graph_triple_tokens(
    groundtruth_cmd_seq, expected
):
    assert (
        StaticLabelDiscreteGraphUpdater.generate_batch_groundtruth_graph_triple_tokens(
            groundtruth_cmd_seq
        )
        == expected
    )


@pytest.mark.parametrize(
    "event_type_ids,src_ids,dst_ids,batch,expected",
    [
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["pad"],
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["end"],
                ]
            ),
            torch.tensor([0, 0, 0]),
            torch.tensor([0, 0, 0]),
            torch.empty(0).long(),
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["pad"],
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["end"],
                ]
            ),
        ),
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                ]
            ),
            torch.tensor([0, 1]),
            torch.tensor([0, 0]),
            torch.tensor([0, 0, 1, 1]),
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                ]
            ),
        ),
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["node-delete"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                ]
            ),
            torch.tensor([2, 0]),
            torch.tensor([0, 1]),
            torch.tensor([0, 0, 0, 1, 1]),
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["node-delete"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                ]
            ),
        ),
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["node-delete"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                ]
            ),
            torch.tensor([3, 5, 2, 0]),
            torch.tensor([0, 0, 4, 1]),
            torch.empty(0).long(),
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["pad"],
                    EVENT_TYPE_ID_MAP["pad"],
                    EVENT_TYPE_ID_MAP["pad"],
                    EVENT_TYPE_ID_MAP["pad"],
                ]
            ),
        ),
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["node-delete"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                ]
            ),
            torch.tensor([3, 5, 2, 0]),
            torch.tensor([0, 0, 4, 1]),
            torch.tensor([3, 3]),
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["pad"],
                    EVENT_TYPE_ID_MAP["pad"],
                    EVENT_TYPE_ID_MAP["pad"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                ]
            ),
        ),
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["node-delete"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                ]
            ),
            torch.tensor([0, 1, 0, 0]),
            torch.tensor([0, 0, 1, 1]),
            torch.tensor([3, 3]),
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["pad"],
                    EVENT_TYPE_ID_MAP["pad"],
                    EVENT_TYPE_ID_MAP["pad"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                ]
            ),
        ),
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["node-delete"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                ]
            ),
            torch.tensor([0, 1, 0, 2]),
            torch.tensor([0, 1, 0, 2]),
            torch.tensor([0, 1, 1, 2, 3, 3, 3]),
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["node-delete"],
                    EVENT_TYPE_ID_MAP["pad"],
                    EVENT_TYPE_ID_MAP["pad"],
                    EVENT_TYPE_ID_MAP["pad"],
                ]
            ),
        ),
    ],
)
def test_sldgu_filter_invalid_events(
    sldgu, event_type_ids, src_ids, dst_ids, batch, expected
):
    assert sldgu.filter_invalid_events(event_type_ids, src_ids, dst_ids, batch).equal(
        expected
    )


@pytest.mark.parametrize(
    "max_decode_len,batch,obs_len,prev_action_len,forward_results,batched_graph,memory,"
    "expected_decoded_list",
    [
        (
            10,
            1,
            3,
            5,
            [
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 1, 0, 0, 0]]
                    ).float(),  # [node-add]
                    "event_src_logits": torch.empty(1, 0),
                    "event_dst_logits": torch.empty(1, 0),
                    "event_label_logits": torch.tensor(
                        [[0, 1, 0, 0, 0, 0]]
                    ).float(),  # [player]
                    "new_decoder_hidden": torch.rand(1, 8),
                    "encoded_obs": torch.rand(1, 3, 8),
                    "encoded_prev_action": torch.rand(1, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.empty(0).long(),
                        x=torch.empty(0, 300),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 300),
                        edge_last_update=torch.empty(0),
                    ),
                    "updated_memory": torch.empty(0, 8),
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 1, 0, 0, 0, 0]]
                    ).float(),  # [end]
                    "event_src_logits": torch.rand(1, 1),
                    "event_dst_logits": torch.rand(1, 1),
                    "event_label_logits": torch.tensor([[1, 0, 0, 0, 0, 0]]).float(),
                    "new_decoder_hidden": torch.rand(1, 8),
                    "encoded_obs": torch.rand(1, 3, 8),
                    "encoded_prev_action": torch.rand(1, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0]),
                        x=torch.tensor([[9] * 300]).float(),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 300),
                        edge_last_update=torch.empty(0),
                    ),
                    "updated_memory": torch.tensor([[3] * 8]).float(),
                },
            ],
            None,
            None,
            [
                {
                    "decoded_event_type_ids": torch.tensor([3]),  # [node-add]
                    "decoded_event_src_ids": torch.zeros(1).long(),
                    "decoded_event_dst_ids": torch.zeros(1).long(),
                    "decoded_event_label_ids": torch.tensor([1]),  # [player]
                    "updated_batched_graph": Batch(
                        batch=torch.empty(0).long(),
                        x=torch.empty(0, 300),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 300),
                        edge_last_update=torch.empty(0),
                    ),
                    "updated_memory": torch.empty(0, 8),
                },
                {
                    "decoded_event_type_ids": torch.tensor([2]),  # [end]
                    "decoded_event_src_ids": torch.zeros(1).long(),
                    "decoded_event_dst_ids": torch.zeros(1).long(),
                    "decoded_event_label_ids": torch.zeros(1).long(),  # [pad]
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0]),
                        x=torch.tensor([[9] * 300]).float(),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 300),
                        edge_last_update=torch.empty(0),
                    ),
                    "updated_memory": torch.tensor([[3] * 8]).float(),
                },
            ],
        ),
        (
            2,
            1,
            3,
            5,
            [
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 1, 0, 0, 0]]
                    ).float(),  # [node-add]
                    "event_src_logits": torch.empty(1, 0),
                    "event_dst_logits": torch.empty(1, 0),
                    "event_label_logits": torch.tensor(
                        [[0, 1, 0, 0, 0, 0]]
                    ).float(),  # [player]
                    "new_decoder_hidden": torch.rand(1, 8),
                    "encoded_obs": torch.rand(1, 3, 8),
                    "encoded_prev_action": torch.rand(1, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.empty(0).long(),
                        x=torch.empty(0, 300),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 300),
                        edge_last_update=torch.empty(0),
                    ),
                    "updated_memory": torch.empty(0, 8),
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 1, 0, 0, 0]]
                    ).float(),  # [node-add]
                    "event_src_logits": torch.rand(1, 1),
                    "event_dst_logits": torch.rand(1, 1),
                    "event_label_logits": torch.tensor(
                        [[0, 0, 1, 0, 0, 0]]
                    ).float(),  # [inventory]
                    "new_decoder_hidden": torch.rand(1, 8),
                    "encoded_obs": torch.rand(1, 3, 8),
                    "encoded_prev_action": torch.rand(1, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0]),
                        x=torch.tensor([[9] * 300]).float(),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 300),
                        edge_last_update=torch.empty(0),
                    ),
                    "updated_memory": torch.tensor([[3] * 8]).float(),
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 0, 0, 1, 0]]
                    ).float(),  # [edge-add]
                    "event_src_logits": torch.tensor([[0, 1]]).float(),
                    "event_dst_logits": torch.tensor([[1, 0]]).float(),
                    "event_label_logits": torch.tensor(
                        [[0, 0, 0, 0, 1, 0]]
                    ).float(),  # [in]
                    "new_decoder_hidden": torch.rand(1, 8),
                    "encoded_obs": torch.rand(1, 3, 8),
                    "encoded_prev_action": torch.rand(1, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0]),
                        x=torch.tensor([[9] * 300, [10] * 300]).float(),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 300),
                        edge_last_update=torch.empty(0),
                    ),
                    "updated_memory": torch.tensor([[3] * 8, [4] * 8]).float(),
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 1, 0, 0, 0, 0]]
                    ).float(),  # [end]
                    "event_src_logits": torch.rand(1, 2),
                    "event_dst_logits": torch.rand(1, 2),
                    "event_label_logits": torch.tensor([[1, 0, 0, 0, 0, 0]]).float(),
                    "new_decoder_hidden": torch.rand(1, 8),
                    "encoded_obs": torch.rand(1, 3, 8),
                    "encoded_prev_action": torch.rand(1, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0]),
                        x=torch.tensor([[9] * 300, [10] * 300]).float(),
                        edge_index=torch.tensor([[1], [0]]),
                        edge_attr=torch.tensor([[8] * 300]).float(),
                        edge_last_update=torch.tensor([4.0]),
                    ),
                    "updated_memory": torch.tensor([[2] * 8, [3] * 8]).float(),
                },
            ],
            None,
            None,
            [
                {
                    "decoded_event_type_ids": torch.tensor([3]),  # [node-add]
                    "decoded_event_src_ids": torch.zeros(1).long(),
                    "decoded_event_dst_ids": torch.zeros(1).long(),
                    "decoded_event_label_ids": torch.tensor([1]),  # [player]
                    "updated_batched_graph": Batch(
                        batch=torch.empty(0).long(),
                        x=torch.empty(0, 300),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 300),
                        edge_last_update=torch.empty(0),
                    ),
                    "updated_memory": torch.empty(0, 8),
                },
                {
                    "decoded_event_type_ids": torch.tensor([3]),  # [node-add]
                    "decoded_event_src_ids": torch.zeros(1).long(),
                    "decoded_event_dst_ids": torch.zeros(1).long(),
                    "decoded_event_label_ids": torch.tensor([2]),  # [inventory]
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0]),
                        x=torch.tensor([[9] * 300]).float(),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 300),
                        edge_last_update=torch.empty(0),
                    ),
                    "updated_memory": torch.tensor([[3] * 8]).float(),
                },
            ],
        ),
        (
            10,
            2,
            3,
            5,
            [
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]
                    ).float(),  # [node-add, node-add]
                    "event_src_logits": torch.empty(2, 0),
                    "event_dst_logits": torch.empty(2, 0),
                    "event_label_logits": torch.tensor(
                        [[0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]]
                    ).float(),  # [player, player]
                    "new_decoder_hidden": torch.rand(2, 8),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.empty(0).long(),
                        x=torch.empty(0, 300),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 300),
                        edge_last_update=torch.empty(0),
                    ),
                    "updated_memory": torch.empty(0, 8),
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]
                    ).float(),  # [end, node-add]
                    "event_src_logits": torch.rand(2, 1),
                    "event_dst_logits": torch.rand(2, 1),
                    "event_label_logits": torch.tensor(
                        [[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]]
                    ).float(),  # [player, inventory]
                    "new_decoder_hidden": torch.rand(2, 8),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1]),
                        x=torch.tensor([[9] * 300, [9] * 300]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 300),
                        edge_last_update=torch.empty(0),
                    ),
                    "updated_memory": torch.tensor([[1] * 8, [2] * 8]).float(),
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0]]
                    ).float(),  # [edge-add, edge-add]
                    "event_src_logits": torch.tensor([[0, 1], [1, 0]]).float(),
                    "event_dst_logits": torch.tensor([[0, 1], [0, 1]]).float(),
                    "event_label_logits": torch.tensor(
                        [[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]]
                    ).float(),  # [in, is]
                    "new_decoder_hidden": torch.rand(2, 8),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[9] * 300, [9] * 300, [10] * 300]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 300),
                        edge_last_update=torch.empty(0),
                    ),
                    "updated_memory": torch.tensor([[3] * 8, [4] * 8, [5] * 8]).float(),
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]]
                    ).float(),  # [node-add, end]
                    "event_src_logits": torch.tensor([[0, 1], [0, 1]]).float(),
                    "event_dst_logits": torch.tensor([[0, 1], [1, 0]]).float(),
                    "event_label_logits": torch.tensor(
                        [[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]]
                    ).float(),  # [player, inventory]
                    "new_decoder_hidden": torch.rand(2, 8),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[9] * 300, [9] * 300, [10] * 300]),
                        edge_index=torch.tensor([[1], [2]]),
                        edge_attr=torch.tensor([[3] * 300]),
                        edge_last_update=torch.tensor([4.0]),
                    ),
                    "updated_memory": torch.tensor([[5] * 8, [6] * 8, [7] * 8]).float(),
                },
            ],
            None,
            None,
            [
                {
                    "decoded_event_type_ids": torch.tensor(
                        [3, 3]
                    ),  # [node-add, node-add]
                    "decoded_event_src_ids": torch.zeros(2).long(),
                    "decoded_event_dst_ids": torch.zeros(2).long(),
                    "decoded_event_label_ids": torch.tensor([1, 1]),  # [player, player]
                    "updated_batched_graph": Batch(
                        batch=torch.empty(0).long(),
                        x=torch.empty(0, 300),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 300),
                        edge_last_update=torch.empty(0),
                    ),
                    "updated_memory": torch.empty(0, 8),
                },
                {
                    "decoded_event_type_ids": torch.tensor([2, 3]),  # [end, node-add]
                    "decoded_event_src_ids": torch.zeros(2).long(),
                    "decoded_event_dst_ids": torch.zeros(2).long(),
                    "decoded_event_label_ids": torch.tensor([1, 2]),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1]),
                        x=torch.tensor([[9] * 300, [9] * 300]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 300),
                        edge_last_update=torch.empty(0),
                    ),
                    "updated_memory": torch.tensor([[1] * 8, [2] * 8]).float(),
                },
                {
                    "decoded_event_type_ids": torch.tensor([0, 5]),  # [pad, edge-add]
                    "decoded_event_src_ids": torch.tensor([0, 0]),
                    "decoded_event_dst_ids": torch.tensor([0, 1]),
                    "decoded_event_label_ids": torch.tensor([0, 5]),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[9] * 300, [9] * 300, [10] * 300]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 300),
                        edge_last_update=torch.empty(0),
                    ),
                    "updated_memory": torch.tensor([[3] * 8, [4] * 8, [5] * 8]).float(),
                },
                {
                    "decoded_event_type_ids": torch.tensor([0, 2]),  # [pad, end]
                    "decoded_event_src_ids": torch.tensor([0, 1]),
                    "decoded_event_dst_ids": torch.tensor([0, 0]),
                    "decoded_event_label_ids": torch.tensor([0, 2]),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[9] * 300, [9] * 300, [10] * 300]),
                        edge_index=torch.tensor([[1], [2]]),
                        edge_attr=torch.tensor([[3] * 300]),
                        edge_last_update=torch.tensor([4.0]),
                    ),
                    "updated_memory": torch.tensor([[5] * 8, [6] * 8, [7] * 8]).float(),
                },
            ],
        ),
        (
            2,
            2,
            3,
            5,
            [
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]
                    ).float(),  # [node-add, node-add]
                    "event_src_logits": torch.empty(2, 0),
                    "event_dst_logits": torch.empty(2, 0),
                    "event_label_logits": torch.tensor(
                        [[0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]]
                    ).float(),  # [player, player]
                    "new_decoder_hidden": torch.rand(2, 8),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.empty(0).long(),
                        x=torch.empty(0, 300),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 300),
                        edge_last_update=torch.empty(0),
                    ),
                    "updated_memory": torch.empty(0, 8),
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]
                    ).float(),  # [end, node-add]
                    "event_src_logits": torch.rand(2, 1),
                    "event_dst_logits": torch.rand(2, 1),
                    "event_label_logits": torch.tensor(
                        [[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]]
                    ).float(),  # [player, inventory]
                    "new_decoder_hidden": torch.rand(2, 8),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1]),
                        x=torch.tensor([[9] * 300, [9] * 300]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 300),
                        edge_last_update=torch.empty(0),
                    ),
                    "updated_memory": torch.tensor([[1] * 8, [2] * 8]).float(),
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0]]
                    ).float(),  # [edge-add, edge-add]
                    "event_src_logits": torch.tensor([[0, 1], [1, 0]]).float(),
                    "event_dst_logits": torch.tensor([[0, 1], [0, 1]]).float(),
                    "event_label_logits": torch.tensor(
                        [[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]]
                    ).float(),  # [in, is]
                    "new_decoder_hidden": torch.rand(2, 8),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[9] * 300, [9] * 300, [10] * 300]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 300),
                        edge_last_update=torch.empty(0),
                    ),
                    "updated_memory": torch.tensor([[3] * 8, [4] * 8, [5] * 8]).float(),
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]]
                    ).float(),  # [node-add, end]
                    "event_src_logits": torch.tensor([[0, 1], [0, 1]]).float(),
                    "event_dst_logits": torch.tensor([[0, 1], [1, 0]]).float(),
                    "event_label_logits": torch.tensor(
                        [[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]]
                    ).float(),  # [player, inventory]
                    "new_decoder_hidden": torch.rand(2, 8),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[9] * 300, [9] * 300, [10] * 300]),
                        edge_index=torch.tensor([[1], [2]]),
                        edge_attr=torch.tensor([[3] * 300]),
                        edge_last_update=torch.tensor([4.0]),
                    ),
                    "updated_memory": torch.tensor([[5] * 8, [6] * 8, [7] * 8]).float(),
                },
            ],
            None,
            None,
            [
                {
                    "decoded_event_type_ids": torch.tensor(
                        [3, 3]
                    ),  # [node-add, node-add]
                    "decoded_event_src_ids": torch.zeros(2).long(),
                    "decoded_event_dst_ids": torch.zeros(2).long(),
                    "decoded_event_label_ids": torch.tensor([1, 1]),  # [player, player]
                    "updated_batched_graph": Batch(
                        batch=torch.empty(0).long(),
                        x=torch.empty(0, 300),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 300),
                        edge_last_update=torch.empty(0),
                    ),
                    "updated_memory": torch.empty(0, 8),
                },
                {
                    "decoded_event_type_ids": torch.tensor([2, 3]),  # [end, node-add]
                    "decoded_event_src_ids": torch.zeros(2).long(),
                    "decoded_event_dst_ids": torch.zeros(2).long(),
                    "decoded_event_label_ids": torch.tensor([1, 2]),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1]),
                        x=torch.tensor([[9] * 300, [9] * 300]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 300),
                        edge_last_update=torch.empty(0),
                    ),
                    "updated_memory": torch.tensor([[1] * 8, [2] * 8]).float(),
                },
            ],
        ),
        (
            10,
            2,
            3,
            5,
            [
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]
                    ).float(),  # [node-add, node-add]
                    "event_src_logits": torch.empty(2, 0),
                    "event_dst_logits": torch.empty(2, 0),
                    "event_label_logits": torch.tensor(
                        [[0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]]
                    ).float(),  # [player, player]
                    "new_decoder_hidden": torch.rand(2, 8),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.empty(0).long(),
                        x=torch.empty(0, 300),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 300),
                        edge_last_update=torch.empty(0),
                    ),
                    "updated_memory": torch.empty(0, 8),
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]
                    ).float(),  # [end, node-add]
                    "event_src_logits": torch.rand(2, 1),
                    "event_dst_logits": torch.rand(2, 1),
                    "event_label_logits": torch.tensor(
                        [[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]]
                    ).float(),  # [player, inventory]
                    "new_decoder_hidden": torch.rand(2, 8),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1]),
                        x=torch.tensor([[9] * 300, [9] * 300]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 300),
                        edge_last_update=torch.empty(0),
                    ),
                    "updated_memory": torch.tensor([[1] * 8, [2] * 8]).float(),
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1, 0]]
                    ).float(),  # [edge-delete, edge-add]
                    "event_src_logits": torch.tensor([[0, 1], [0, 1]]).float(),
                    "event_dst_logits": torch.tensor([[1, 0], [1, 0]]).float(),
                    "event_label_logits": torch.tensor(
                        [[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0]]
                    ).float(),  # [is, in]
                    "new_decoder_hidden": torch.rand(2, 8),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[9] * 300, [9] * 300, [10] * 300]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 300),
                        edge_last_update=torch.empty(0),
                    ),
                    "updated_memory": torch.tensor([[3] * 8, [4] * 8, [5] * 8]).float(),
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1]]
                    ).float(),  # [edge-delete, edge-delete]
                    "event_src_logits": torch.tensor([[1, 0, 0], [0, 0, 1]]).float(),
                    "event_dst_logits": torch.tensor([[0, 0, 1], [0, 1, 0]]).float(),
                    "event_label_logits": torch.tensor(
                        [[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0]]
                    ).float(),  # [is, in]
                    "new_decoder_hidden": torch.rand(2, 8),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[9] * 300, [9] * 300, [10] * 300]),
                        edge_index=torch.tensor([[2], [1]]),
                        edge_attr=torch.tensor([[8] * 300]),
                        edge_last_update=torch.tensor([4.0]),
                    ),
                    "updated_memory": torch.tensor([[2] * 8, [3] * 8, [4] * 8]).float(),
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0, 0]]
                    ).float(),  # [edge-delete, node-delete]
                    "event_src_logits": torch.tensor([[0, 0, 1], [0, 0, 1]]).float(),
                    "event_dst_logits": torch.tensor([[1, 0, 0], [0, 0, 1]]).float(),
                    "event_label_logits": torch.tensor(
                        [[0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]]
                    ).float(),  # [player, player]
                    "new_decoder_hidden": torch.rand(2, 8),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[9] * 300, [9] * 300, [10] * 300]),
                        edge_index=torch.tensor([[2], [1]]),
                        edge_attr=torch.tensor([[8] * 300]),
                        edge_last_update=torch.tensor([4.0]),
                    ),
                    "updated_memory": torch.tensor([[2] * 8, [3] * 8, [4] * 8]).float(),
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]]
                    ).float(),  # [node-add, end]
                    "event_src_logits": torch.tensor([[0, 0, 1], [0, 1, 0]]).float(),
                    "event_dst_logits": torch.tensor([[1, 0, 0], [0, 0, 1]]).float(),
                    "event_label_logits": torch.tensor(
                        [[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]]
                    ).float(),  # [player, inventory]
                    "new_decoder_hidden": torch.rand(2, 8),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[9] * 300, [9] * 300, [10] * 300]),
                        edge_index=torch.tensor([[2], [1]]),
                        edge_attr=torch.tensor([[8] * 300]),
                        edge_last_update=torch.tensor([4.0]),
                    ),
                    "updated_memory": torch.tensor([[2] * 8, [3] * 8, [4] * 8]).float(),
                },
            ],
            None,
            None,
            [
                {
                    "decoded_event_type_ids": torch.tensor(
                        [3, 3]
                    ),  # [node-add, node-add]
                    "decoded_event_src_ids": torch.zeros(2).long(),
                    "decoded_event_dst_ids": torch.zeros(2).long(),
                    "decoded_event_label_ids": torch.tensor([1, 1]),  # [player, player]
                    "updated_batched_graph": Batch(
                        batch=torch.empty(0).long(),
                        x=torch.empty(0, 300),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 300),
                        edge_last_update=torch.empty(0),
                    ),
                    "updated_memory": torch.empty(0, 8),
                },
                {
                    "decoded_event_type_ids": torch.tensor([2, 3]),  # [end, node-add]
                    "decoded_event_src_ids": torch.tensor([0, 0]),
                    "decoded_event_dst_ids": torch.tensor([0, 0]),
                    "decoded_event_label_ids": torch.tensor(
                        [1, 2]
                    ),  # [inventory, inventory]
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1]),
                        x=torch.tensor([[9] * 300, [9] * 300]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 300),
                        edge_last_update=torch.empty(0),
                    ),
                    "updated_memory": torch.tensor([[1] * 8, [2] * 8]).float(),
                },
                {
                    "decoded_event_type_ids": torch.tensor([0, 5]),  # [pad, edge-add]
                    "decoded_event_src_ids": torch.tensor([0, 1]),
                    "decoded_event_dst_ids": torch.tensor([0, 0]),
                    "decoded_event_label_ids": torch.tensor([0, 4]),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[9] * 300, [9] * 300, [10] * 300]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 300),
                        edge_last_update=torch.empty(0),
                    ),
                    "updated_memory": torch.tensor([[3] * 8, [4] * 8, [5] * 8]).float(),
                },
                {
                    "decoded_event_type_ids": torch.tensor([0, 0]),  # [pad, pad]
                    "decoded_event_src_ids": torch.tensor([0, 2]),
                    "decoded_event_dst_ids": torch.tensor([0, 1]),
                    "decoded_event_label_ids": torch.tensor([0, 4]),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[9] * 300, [9] * 300, [10] * 300]),
                        edge_index=torch.tensor([[2], [1]]),
                        edge_attr=torch.tensor([[8] * 300]),
                        edge_last_update=torch.tensor([4.0]),
                    ),
                    "updated_memory": torch.tensor([[2] * 8, [3] * 8, [4] * 8]).float(),
                },
                {
                    "decoded_event_type_ids": torch.tensor([0, 0]),  # [pad, pad]
                    "decoded_event_src_ids": torch.tensor([0, 2]),
                    "decoded_event_dst_ids": torch.tensor([0, 2]),
                    "decoded_event_label_ids": torch.tensor([0, 1]),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[9] * 300, [9] * 300, [10] * 300]),
                        edge_index=torch.tensor([[2], [1]]),
                        edge_attr=torch.tensor([[8] * 300]),
                        edge_last_update=torch.tensor([4.0]),
                    ),
                    "updated_memory": torch.tensor([[2] * 8, [3] * 8, [4] * 8]).float(),
                },
                {
                    "decoded_event_type_ids": torch.tensor([0, 2]),  # [pad, end]
                    "decoded_event_src_ids": torch.tensor([0, 1]),
                    "decoded_event_dst_ids": torch.tensor([0, 2]),
                    "decoded_event_label_ids": torch.tensor([0, 2]),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[9] * 300, [9] * 300, [10] * 300]),
                        edge_index=torch.tensor([[2], [1]]),
                        edge_attr=torch.tensor([[8] * 300]),
                        edge_last_update=torch.tensor([4.0]),
                    ),
                    "updated_memory": torch.tensor([[2] * 8, [3] * 8, [4] * 8]).float(),
                },
            ],
        ),
        (
            10,
            2,
            3,
            5,
            [
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]
                    ).float(),  # [node-add, node-add]
                    "event_src_logits": torch.tensor([[0, 1], [1, 0]]).float(),
                    "event_dst_logits": torch.tensor([[0, 1], [0, 1]]).float(),
                    "event_label_logits": torch.tensor(
                        [[0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]]
                    ).float(),  # [player, player]
                    "new_decoder_hidden": torch.rand(2, 8),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[9] * 300, [9] * 300, [10] * 300]),
                        edge_index=torch.tensor([[1], [2]]),
                        edge_attr=torch.tensor([[3] * 300]),
                        edge_last_update=torch.tensor([4.0]),
                    ),
                    "updated_memory": torch.tensor([[5] * 8, [6] * 8, [7] * 8]).float(),
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]
                    ).float(),  # [end, node-add]
                    "event_src_logits": torch.tensor([[0, 1, 0], [0, 0, 1]]).float(),
                    "event_dst_logits": torch.tensor([[0, 1, 0], [0, 0, 1]]).float(),
                    "event_label_logits": torch.tensor(
                        [[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]]
                    ).float(),  # [player, inventory]
                    "new_decoder_hidden": torch.rand(2, 8),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1]),
                        x=torch.tensor(
                            [[9] * 300, [9] * 300, [9] * 300, [10] * 300, [9] * 300]
                        ),
                        edge_index=torch.tensor([[2], [3]]),
                        edge_attr=torch.tensor([[3] * 300]),
                        edge_last_update=torch.tensor([4.0]),
                    ),
                    "updated_memory": torch.tensor(
                        [[5] * 8, [6] * 8, [7] * 8, [8] * 8, [9] * 8]
                    ).float(),
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0]]
                    ).float(),  # [edge-add, edge-add]
                    "event_src_logits": torch.tensor(
                        [[0, 1, 0, 0], [0, 0, 1, 0]]
                    ).float(),
                    "event_dst_logits": torch.tensor(
                        [[0, 1, 0, 0], [0, 0, 0, 1]]
                    ).float(),
                    "event_label_logits": torch.tensor(
                        [[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]]
                    ).float(),  # [in, is]
                    "new_decoder_hidden": torch.rand(2, 8),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1, 1]),
                        x=torch.tensor(
                            [
                                [9] * 300,
                                [9] * 300,
                                [9] * 300,
                                [10] * 300,
                                [9] * 300,
                                [10] * 300,
                            ]
                        ),
                        edge_index=torch.tensor([[2], [3]]),
                        edge_attr=torch.tensor([[3] * 300]),
                        edge_last_update=torch.tensor([4.0]),
                    ),
                    "updated_memory": torch.tensor(
                        [[3] * 8, [5] * 8, [6] * 8, [7] * 8, [8] * 8, [9] * 8]
                    ).float(),
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]]
                    ).float(),  # [node-add, end]
                    "event_src_logits": torch.tensor(
                        [[0, 1, 0, 0], [0, 0, 1, 0]]
                    ).float(),
                    "event_dst_logits": torch.tensor(
                        [[0, 1, 0, 0], [0, 0, 0, 1]]
                    ).float(),
                    "event_label_logits": torch.tensor(
                        [[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]]
                    ).float(),  # [player, inventory]
                    "new_decoder_hidden": torch.rand(2, 8),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1, 1]),
                        x=torch.tensor(
                            [
                                [9] * 300,
                                [9] * 300,
                                [9] * 300,
                                [10] * 300,
                                [9] * 300,
                                [10] * 300,
                            ]
                        ),
                        edge_index=torch.tensor([[2, 5], [3, 6]]),
                        edge_attr=torch.tensor([[3] * 300, [3] * 300]),
                        edge_last_update=torch.tensor([4.0, 5.0]),
                    ),
                    "updated_memory": torch.tensor(
                        [[3] * 8, [5] * 8, [6] * 8, [7] * 8, [8] * 8, [9] * 8]
                    ).float(),
                },
            ],
            Batch(
                batch=torch.tensor([0, 1, 1]),
                x=torch.tensor([[9] * 300, [9] * 300, [10] * 300]),
                edge_index=torch.tensor([[1], [2]]),
                edge_attr=torch.tensor([[3] * 300]),
                edge_last_update=torch.tensor([4.0]),
            ),
            torch.tensor([[5] * 8, [6] * 8, [7] * 8]).float(),
            [
                {
                    "decoded_event_type_ids": torch.tensor(
                        [3, 3]
                    ),  # [node-add, node-add]
                    "decoded_event_src_ids": torch.tensor([1, 0]),
                    "decoded_event_dst_ids": torch.tensor([1, 1]),
                    "decoded_event_label_ids": torch.tensor([1, 1]),  # [player, player]
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[9] * 300, [9] * 300, [10] * 300]),
                        edge_index=torch.tensor([[1], [2]]),
                        edge_attr=torch.tensor([[3] * 300]),
                        edge_last_update=torch.tensor([4.0]),
                    ),
                    "updated_memory": torch.tensor([[5] * 8, [6] * 8, [7] * 8]).float(),
                },
                {
                    "decoded_event_type_ids": torch.tensor([2, 3]),  # [end, node-add]
                    "decoded_event_src_ids": torch.tensor([1, 2]),
                    "decoded_event_dst_ids": torch.tensor([1, 2]),
                    "decoded_event_label_ids": torch.tensor([1, 2]),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1]),
                        x=torch.tensor(
                            [[9] * 300, [9] * 300, [9] * 300, [10] * 300, [9] * 300]
                        ),
                        edge_index=torch.tensor([[2], [3]]),
                        edge_attr=torch.tensor([[3] * 300]),
                        edge_last_update=torch.tensor([4.0]),
                    ),
                    "updated_memory": torch.tensor(
                        [[5] * 8, [6] * 8, [7] * 8, [8] * 8, [9] * 8]
                    ).float(),
                },
                {
                    "decoded_event_type_ids": torch.tensor([0, 5]),  # [pad, edge-add]
                    "decoded_event_src_ids": torch.tensor([0, 2]),
                    "decoded_event_dst_ids": torch.tensor([0, 3]),
                    "decoded_event_label_ids": torch.tensor([0, 5]),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1, 1]),
                        x=torch.tensor(
                            [
                                [9] * 300,
                                [9] * 300,
                                [9] * 300,
                                [10] * 300,
                                [9] * 300,
                                [10] * 300,
                            ]
                        ),
                        edge_index=torch.tensor([[2], [3]]),
                        edge_attr=torch.tensor([[3] * 300]),
                        edge_last_update=torch.tensor([4.0]),
                    ),
                    "updated_memory": torch.tensor(
                        [[3] * 8, [5] * 8, [6] * 8, [7] * 8, [8] * 8, [9] * 8]
                    ).float(),
                },
                {
                    "decoded_event_type_ids": torch.tensor([0, 2]),  # [pad, end]
                    "decoded_event_src_ids": torch.tensor([0, 2]),
                    "decoded_event_dst_ids": torch.tensor([0, 3]),
                    "decoded_event_label_ids": torch.tensor([0, 2]),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1, 1]),
                        x=torch.tensor(
                            [
                                [9] * 300,
                                [9] * 300,
                                [9] * 300,
                                [10] * 300,
                                [9] * 300,
                                [10] * 300,
                            ]
                        ),
                        edge_index=torch.tensor([[2, 5], [3, 6]]),
                        edge_attr=torch.tensor([[3] * 300, [3] * 300]),
                        edge_last_update=torch.tensor([4.0, 5.0]),
                    ),
                    "updated_memory": torch.tensor(
                        [[3] * 8, [5] * 8, [6] * 8, [7] * 8, [8] * 8, [9] * 8]
                    ).float(),
                },
            ],
        ),
        (
            2,
            2,
            3,
            5,
            [
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]
                    ).float(),  # [node-add, node-add]
                    "event_src_logits": torch.tensor([[0, 1], [1, 0]]).float(),
                    "event_dst_logits": torch.tensor([[0, 1], [0, 1]]).float(),
                    "event_label_logits": torch.tensor(
                        [[0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]]
                    ).float(),  # [player, player]
                    "new_decoder_hidden": torch.rand(2, 8),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[9] * 300, [9] * 300, [10] * 300]),
                        edge_index=torch.tensor([[1], [2]]),
                        edge_attr=torch.tensor([[3] * 300]),
                        edge_last_update=torch.tensor([4.0]),
                    ),
                    "updated_memory": torch.tensor([[5] * 8, [6] * 8, [7] * 8]).float(),
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]
                    ).float(),  # [end, node-add]
                    "event_src_logits": torch.tensor([[0, 1, 0], [0, 0, 1]]).float(),
                    "event_dst_logits": torch.tensor([[0, 1, 0], [0, 0, 1]]).float(),
                    "event_label_logits": torch.tensor(
                        [[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]]
                    ).float(),  # [player, inventory]
                    "new_decoder_hidden": torch.rand(2, 8),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1]),
                        x=torch.tensor(
                            [[9] * 300, [9] * 300, [9] * 300, [10] * 300, [9] * 300]
                        ),
                        edge_index=torch.tensor([[2], [3]]),
                        edge_attr=torch.tensor([[3] * 300]),
                        edge_last_update=torch.tensor([4.0]),
                    ),
                    "updated_memory": torch.tensor(
                        [[5] * 8, [6] * 8, [7] * 8, [8] * 8, [9] * 8]
                    ).float(),
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0]]
                    ).float(),  # [edge-add, edge-add]
                    "event_src_logits": torch.tensor(
                        [[0, 1, 0, 0], [0, 0, 1, 0]]
                    ).float(),
                    "event_dst_logits": torch.tensor(
                        [[0, 1, 0, 0], [0, 0, 0, 1]]
                    ).float(),
                    "event_label_logits": torch.tensor(
                        [[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]]
                    ).float(),  # [in, is]
                    "new_decoder_hidden": torch.rand(2, 8),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1, 1]),
                        x=torch.tensor(
                            [
                                [9] * 300,
                                [9] * 300,
                                [9] * 300,
                                [10] * 300,
                                [9] * 300,
                                [10] * 300,
                            ]
                        ),
                        edge_index=torch.tensor([[2], [3]]),
                        edge_attr=torch.tensor([[3] * 300]),
                        edge_last_update=torch.tensor([4.0]),
                    ),
                    "updated_memory": torch.tensor(
                        [[3] * 8, [5] * 8, [6] * 8, [7] * 8, [8] * 8, [9] * 8]
                    ).float(),
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]]
                    ).float(),  # [node-add, end]
                    "event_src_logits": torch.tensor(
                        [[0, 1, 0, 0], [0, 0, 1, 0]]
                    ).float(),
                    "event_dst_logits": torch.tensor(
                        [[0, 1, 0, 0], [0, 0, 0, 1]]
                    ).float(),
                    "event_label_logits": torch.tensor(
                        [[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]]
                    ).float(),  # [player, inventory]
                    "new_decoder_hidden": torch.rand(2, 8),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1, 1]),
                        x=torch.tensor(
                            [
                                [9] * 300,
                                [9] * 300,
                                [9] * 300,
                                [10] * 300,
                                [9] * 300,
                                [10] * 300,
                            ]
                        ),
                        edge_index=torch.tensor([[2, 5], [3, 6]]),
                        edge_attr=torch.tensor([[3] * 300, [3] * 300]),
                        edge_last_update=torch.tensor([4.0, 5.0]),
                    ),
                    "updated_memory": torch.tensor(
                        [[3] * 8, [5] * 8, [6] * 8, [7] * 8, [8] * 8, [9] * 8]
                    ).float(),
                },
            ],
            Batch(
                batch=torch.tensor([0, 1, 1]),
                x=torch.tensor([[9] * 300, [9] * 300, [10] * 300]),
                edge_index=torch.tensor([[1], [2]]),
                edge_attr=torch.tensor([[3] * 300]),
                edge_last_update=torch.tensor([4.0]),
            ),
            torch.tensor([[5] * 8, [6] * 8, [7] * 8]).float(),
            [
                {
                    "decoded_event_type_ids": torch.tensor(
                        [3, 3]
                    ),  # [node-add, node-add]
                    "decoded_event_src_ids": torch.tensor([1, 0]),
                    "decoded_event_dst_ids": torch.tensor([1, 1]),
                    "decoded_event_label_ids": torch.tensor([1, 1]),  # [player, player]
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[9] * 300, [9] * 300, [10] * 300]),
                        edge_index=torch.tensor([[1], [2]]),
                        edge_attr=torch.tensor([[3] * 300]),
                        edge_last_update=torch.tensor([4.0]),
                    ),
                    "updated_memory": torch.tensor([[5] * 8, [6] * 8, [7] * 8]).float(),
                },
                {
                    "decoded_event_type_ids": torch.tensor([2, 3]),  # [end, node-add]
                    "decoded_event_src_ids": torch.tensor([1, 2]),
                    "decoded_event_dst_ids": torch.tensor([1, 2]),
                    "decoded_event_label_ids": torch.tensor([1, 2]),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1]),
                        x=torch.tensor(
                            [[9] * 300, [9] * 300, [9] * 300, [10] * 300, [9] * 300]
                        ),
                        edge_index=torch.tensor([[2], [3]]),
                        edge_attr=torch.tensor([[3] * 300]),
                        edge_last_update=torch.tensor([4.0]),
                    ),
                    "updated_memory": torch.tensor(
                        [[5] * 8, [6] * 8, [7] * 8, [8] * 8, [9] * 8]
                    ).float(),
                },
            ],
        ),
    ],
)
def test_sldgu_greedy_decode(
    monkeypatch,
    sldgu,
    max_decode_len,
    batch,
    obs_len,
    prev_action_len,
    forward_results,
    batched_graph,
    memory,
    expected_decoded_list,
):
    class MockForward:
        def __init__(self):
            self.num_calls = 0

        def __call__(self, *args, **kwargs):
            decoded = forward_results[self.num_calls]
            decoded["new_decoder_hidden"] = torch.rand(batch, sldgu.hparams.hidden_dim)
            decoded["encoded_obs"] = torch.rand(
                batch, obs_len, sldgu.hparams.hidden_dim
            )
            decoded["encoded_prev_action"] = torch.rand(
                batch, prev_action_len, sldgu.hparams.hidden_dim
            )
            self.num_calls += 1
            return decoded

    monkeypatch.setattr(sldgu, "forward", MockForward())
    monkeypatch.setattr(sldgu.hparams, "max_decode_len", max_decode_len)
    decoded_list = sldgu.greedy_decode(
        TWCmdGenTemporalStepInput(
            obs_word_ids=torch.randint(
                len(sldgu.preprocessor.word_vocab), (batch, obs_len)
            ),
            obs_mask=torch.randint(2, (batch, obs_len)).float(),
            prev_action_word_ids=torch.randint(
                len(sldgu.preprocessor.word_vocab), (batch, prev_action_len)
            ),
            prev_action_mask=torch.randint(2, (batch, prev_action_len)).float(),
            timestamps=torch.tensor([4.0] * batch),
        ),
        batched_graph=batched_graph,
        memory=memory,
    )

    assert len(decoded_list) == len(expected_decoded_list)
    for decoded, expected in zip(decoded_list, expected_decoded_list):
        assert decoded["decoded_event_type_ids"].equal(
            expected["decoded_event_type_ids"]
        )
        assert decoded["decoded_event_src_ids"].equal(expected["decoded_event_src_ids"])
        assert decoded["decoded_event_dst_ids"].equal(expected["decoded_event_dst_ids"])
        assert decoded["decoded_event_label_ids"].equal(
            expected["decoded_event_label_ids"]
        )
        assert decoded["updated_batched_graph"].batch.equal(
            expected["updated_batched_graph"].batch
        )
        assert decoded["updated_batched_graph"].x.equal(
            expected["updated_batched_graph"].x
        )
        assert decoded["updated_batched_graph"].edge_index.equal(
            expected["updated_batched_graph"].edge_index
        )
        assert decoded["updated_batched_graph"].edge_attr.equal(
            expected["updated_batched_graph"].edge_attr
        )
        assert decoded["updated_batched_graph"].edge_last_update.equal(
            expected["updated_batched_graph"].edge_last_update
        )
        assert decoded["updated_memory"].equal(expected["updated_memory"])


@pytest.mark.parametrize(
    "ids,timestamps,mask,batch_cmds,expected",
    [
        (
            (("g0", 0),),
            torch.tensor([1.0]),
            [True],
            [
                (("add , player , kitchen , in", "add , table , kitchen , in"),),
                [["add , player , kitchen , in", "add , fire , kitchen , in"]],
                [["add , player , kitchen , in", "delete , player , kitchen , in"]],
            ],
            [
                (
                    "g0|0|1",
                    "add , player , kitchen , in | add , table , kitchen , in",
                    "add , player , kitchen , in | add , fire , kitchen , in",
                    "add , player , kitchen , in | delete , player , kitchen , in",
                )
            ],
        ),
        (
            (("g0", 0), ("g1", 2), ("g2", 1)),
            torch.tensor([1.0, 2.0, 3.0]),
            [True, False, True],
            [
                (
                    ("add , player , kitchen , in", "add , table , kitchen , in"),
                    (),
                    ("add , player , livingroom , in", "add , sofa , livingroom , in"),
                ),
                [
                    ["add , player , kitchen , in", "add , fire , kitchen , in"],
                    ["add , player , garage , in"],
                    ["add , player , livingroom , in", "add , TV , livingroom , in"],
                ],
                [
                    ["add , player , kitchen , in", "delete , player , kitchen , in"],
                    ["add , desk , garage , in"],
                    ["add , player , garage , in", "add , sofa , livingroom , in"],
                ],
            ],
            [
                (
                    "g0|0|1",
                    "add , player , kitchen , in | add , table , kitchen , in",
                    "add , player , kitchen , in | add , fire , kitchen , in",
                    "add , player , kitchen , in | delete , player , kitchen , in",
                ),
                (
                    "g2|1|3",
                    "add , player , livingroom , in | add , sofa , livingroom , in",
                    "add , player , livingroom , in | add , TV , livingroom , in",
                    "add , player , garage , in | add , sofa , livingroom , in",
                ),
            ],
        ),
    ],
)
def test_sldgu_generate_predict_table_rows(ids, timestamps, mask, batch_cmds, expected):
    assert (
        StaticLabelDiscreteGraphUpdater.generate_predict_table_rows(
            ids, timestamps, mask, *batch_cmds
        )
        == expected
    )
