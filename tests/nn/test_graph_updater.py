import pytest
import torch

from dgu.nn.graph_updater import StaticLabelDiscreteGraphUpdater
from dgu.constants import EVENT_TYPES, EVENT_TYPE_ID_MAP


@pytest.fixture
def sldgu():
    return StaticLabelDiscreteGraphUpdater(
        pretrained_word_embedding_path="tests/data/test-fasttext.vec",
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
    results = sldgu(
        torch.randint(2, (batch_size, obs_len)).float(),
        torch.randint(2, (batch_size, prev_action_len)).float(),
        event_type_ids,
        event_src_ids,
        event_dst_ids,
        torch.randint(len(sldgu.labels), (batch_size,)),
        torch.randint(10, (batch_size,)).float(),
        torch.rand(prev_num_node, sldgu.tgn.memory_dim),
        torch.rand(num_node, sldgu.tgn.event_embedding_dim),
        torch.randint(num_node, (prev_num_node,)),
        torch.randint(2, (prev_num_node,)),
        torch.randint(num_node, (2, num_edge)),
        torch.rand(num_edge, sldgu.tgn.event_embedding_dim),
        torch.randint(10, (num_edge,)),
        torch.randint(10, (num_edge,)).float(),
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
    max_sub_graph_num_node = batch.bincount().max().item()
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
    "event_type_ids,src_ids,dst_ids,event_label_ids,event_timestamps,expected",
    [
        (
            torch.tensor([EVENT_TYPE_ID_MAP["start"], EVENT_TYPE_ID_MAP["start"]]),
            torch.tensor([0, 0]),
            torch.tensor([0, 0]),
            torch.tensor([0, 0]),
            torch.tensor([0, 0]).float(),
            {
                "edge_event_type_ids": torch.empty(0).long(),
                "edge_event_src_ids": torch.empty(0).long(),
                "edge_event_dst_ids": torch.empty(0).long(),
                "edge_event_label_ids": torch.empty(0).long(),
                "edge_event_timestamps": torch.empty(0),
            },
        ),
        (
            torch.tensor(
                [
                    EVENT_TYPE_ID_MAP["start"],
                    EVENT_TYPE_ID_MAP["node-add"],
                    EVENT_TYPE_ID_MAP["edge-add"],
                    EVENT_TYPE_ID_MAP["node-delete"],
                    EVENT_TYPE_ID_MAP["edge-delete"],
                    EVENT_TYPE_ID_MAP["end"],
                ]
            ),
            torch.tensor([0, 0, 4, 2, 3, 0]),
            torch.tensor([0, 0, 1, 0, 5, 0]),
            torch.tensor([0, 2, 4, 2, 6, 0]),
            torch.tensor([0, 4, 6, 2, 4, 0]).float(),
            {
                "edge_event_type_ids": torch.tensor(
                    [EVENT_TYPE_ID_MAP["edge-add"], EVENT_TYPE_ID_MAP["edge-delete"]]
                ),
                "edge_event_src_ids": torch.tensor([4, 3]),
                "edge_event_dst_ids": torch.tensor([1, 5]),
                "edge_event_label_ids": torch.tensor([4, 6]),
                "edge_event_timestamps": torch.tensor([6.0, 4.0]),
            },
        ),
    ],
)
def test_sldgu_get_edge_events(
    event_type_ids, src_ids, dst_ids, event_label_ids, event_timestamps, expected
):
    results = StaticLabelDiscreteGraphUpdater.get_edge_events(
        event_type_ids, src_ids, dst_ids, event_label_ids, event_timestamps
    )
    for k in [
        "edge_event_type_ids",
        "edge_event_src_ids",
        "edge_event_dst_ids",
        "edge_event_label_ids",
        "edge_event_timestamps",
    ]:
        assert results[k].equal(expected[k])


@pytest.mark.parametrize(
    "node_embeddings,batch,expected_batch_node_embeddings,expected_batch_mask",
    [
        (
            torch.tensor(
                [[2] * 4, [3] * 4, [4] * 4, [5] * 4, [6] * 4, [7] * 4]
            ).float(),
            torch.tensor([0, 1, 1, 2, 2, 2]),
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
            torch.tensor(
                [
                    [[2] * 4, [3] * 4, [4] * 4],
                    [[5] * 4, [0] * 4, [0] * 4],
                    [[6] * 4, [7] * 4, [0] * 4],
                ]
            ).float(),
            torch.tensor([[1, 1, 1], [1, 0, 0], [1, 1, 0]]).float(),
        ),
    ],
)
def test_sldgu_batchify_node_embeddings(
    node_embeddings, batch, expected_batch_node_embeddings, expected_batch_mask
):
    (
        batch_node_embeddings,
        batch_mask,
    ) = StaticLabelDiscreteGraphUpdater.batchify_node_embeddings(node_embeddings, batch)
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
    "event_type_ids,src_ids,dst_ids,label_ids,node_label_ids,batch,expected",
    [
        (
            torch.tensor([EVENT_TYPE_ID_MAP["node-add"]]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([0]),  # player
            torch.tensor([0]),
            torch.tensor([0]),
            ([], []),
        ),
        (
            torch.tensor([EVENT_TYPE_ID_MAP["node-delete"]]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([0]),  # player
            torch.tensor([0]),
            torch.tensor([0]),
            ([], []),
        ),
        (
            torch.tensor([EVENT_TYPE_ID_MAP["edge-add"]]),
            torch.tensor([1]),
            torch.tensor([0]),
            torch.tensor([5]),  # is
            torch.tensor([3, 1]),
            torch.tensor([0, 0]),
            (["add , player , chopped , is"], [["add", "player", "chopped", "is"]]),
        ),
        (
            torch.tensor([EVENT_TYPE_ID_MAP["edge-delete"]]),
            torch.tensor([1]),
            torch.tensor([0]),
            torch.tensor([5]),  # is
            torch.tensor([3, 1]),
            torch.tensor([0, 0]),
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
            torch.tensor([3, 1, 1, 2, 1, 2]),
            torch.tensor([0, 0, 0, 0, 1, 1]),
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
    node_label_ids,
    batch,
    expected,
):
    assert (
        sldgu.generate_graph_triples(
            event_type_ids,
            src_ids,
            dst_ids,
            label_ids,
            node_label_ids,
            batch,
        )
        == expected
    )


@pytest.mark.parametrize(
    "event_type_id_seq,src_id_seq,dst_id_seq,label_id_seq,node_label_id_seq,batch_seq,"
    "expected",
    [
        (
            [torch.tensor([EVENT_TYPE_ID_MAP["node-add"]])],
            [torch.tensor([0])],
            [torch.tensor([0])],
            [torch.tensor([0])],
            [torch.tensor([0])],
            [torch.tensor([0])],
            ([[]], [[]]),
        ),
        (
            [torch.tensor([EVENT_TYPE_ID_MAP["node-delete"]])],
            [torch.tensor([0])],
            [torch.tensor([0])],
            [torch.tensor([0])],
            [torch.tensor([0])],
            [torch.tensor([0])],
            ([[]], [[]]),
        ),
        (
            [torch.tensor([EVENT_TYPE_ID_MAP["edge-add"]])],
            [torch.tensor([1])],
            [torch.tensor([0])],
            [torch.tensor([5])],
            [torch.tensor([3, 1])],
            [torch.tensor([0, 0])],
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
            [torch.tensor([3, 1])],
            [torch.tensor([0, 0])],
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
            [torch.tensor([3, 1, 1, 2, 1, 2]), torch.tensor([1, 3, 1, 2, 1, 2])],
            [torch.tensor([0, 0, 0, 0, 1, 1]), torch.tensor([0, 0, 1, 1, 1, 1])],
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
    node_label_id_seq,
    batch_seq,
    expected,
):
    assert (
        sldgu.generate_batch_graph_triples_seq(
            event_type_id_seq,
            src_id_seq,
            dst_id_seq,
            label_id_seq,
            node_label_id_seq,
            batch_seq,
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
