import pytest
import torch

from dgu.nn.graph_updater import StaticLabelDiscreteGraphUpdater
from dgu.constants import EVENT_TYPES


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
    "batch,obs_len,prev_action_len,prev_num_node,prev_num_edge,"
    "prev_graph_event_seq_len",
    [
        (1, 10, 5, 0, 0, 0),
        (1, 10, 5, 7, 9, 1),
        (1, 10, 5, 7, 9, 4),
        (4, 12, 8, 6, 12, 1),
        (4, 12, 8, 6, 12, 10),
    ],
)
@pytest.mark.parametrize("encoded_textual_input", [True, False])
@pytest.mark.parametrize("decoder_hidden", [True, False])
def test_sldgu_forward(
    sldgu,
    decoder_hidden,
    encoded_textual_input,
    batch,
    obs_len,
    prev_action_len,
    prev_num_node,
    prev_num_edge,
    prev_graph_event_seq_len,
):
    num_label = (
        sldgu.decoder.graph_event_decoder.event_label_head.num_node_label
        + sldgu.decoder.graph_event_decoder.event_label_head.num_edge_label
    )
    encoded_obs = (
        torch.rand(batch, obs_len, sldgu.hparams.hidden_dim)
        if encoded_textual_input
        else None
    )
    encoded_prev_action = (
        torch.rand(batch, prev_action_len, sldgu.hparams.hidden_dim)
        if encoded_textual_input
        else None
    )
    results = sldgu(
        torch.randint(2, (batch, obs_len)).float(),
        torch.randint(2, (batch, prev_action_len)).float(),
        torch.randint(len(EVENT_TYPES), (batch, prev_graph_event_seq_len)),
        torch.randint(prev_num_node, (batch, prev_graph_event_seq_len))
        if prev_num_node > 0
        else torch.zeros(batch, 0).long(),
        torch.randint(2, (batch, prev_graph_event_seq_len)).float(),
        torch.randint(prev_num_node, (batch, prev_graph_event_seq_len))
        if prev_num_node > 0
        else torch.zeros(batch, 0).long(),
        torch.randint(2, (batch, prev_graph_event_seq_len)).float(),
        torch.randint(prev_num_edge, (batch, prev_graph_event_seq_len))
        if prev_num_edge > 0
        else torch.zeros(batch, 0).long(),
        torch.randint(num_label, (batch, prev_graph_event_seq_len)),
        torch.randint(10, (batch, prev_graph_event_seq_len)).float(),
        torch.randint(prev_num_node, (batch, prev_num_node))
        if prev_num_node > 0
        else torch.zeros(batch, 0).long(),
        torch.randint(2, (batch, prev_num_node)).float(),
        torch.randint(prev_num_edge, (batch, prev_num_edge))
        if prev_num_edge > 0
        else torch.zeros(batch, 0).long(),
        torch.randint(prev_num_node, (batch, 2, prev_num_edge))
        if prev_num_node > 0
        else torch.zeros(batch, 2, 0).long(),
        torch.randint(10, (batch, prev_num_edge)),
        decoder_hidden=torch.rand(batch, sldgu.hparams.hidden_dim)
        if decoder_hidden
        else None,
        obs_word_ids=None
        if encoded_textual_input
        else torch.randint(len(sldgu.preprocessor.word_to_id_dict), (batch, obs_len)),
        prev_action_word_ids=None
        if encoded_textual_input
        else torch.randint(
            len(sldgu.preprocessor.word_to_id_dict), (batch, prev_action_len)
        ),
        encoded_obs=encoded_obs,
        encoded_prev_action=encoded_prev_action,
    )
    assert results["event_type_logits"].size() == (batch, len(EVENT_TYPES))
    assert results["src_logits"].size() == (batch, prev_num_node)
    assert results["dst_logits"].size() == (batch, prev_num_node)
    assert results["label_logits"].size() == (batch, num_label)
    assert results["new_hidden"].size() == (batch, sldgu.hparams.hidden_dim)
    if encoded_textual_input:
        assert results["encoded_obs"].equal(encoded_obs)
        assert results["encoded_prev_action"].equal(encoded_prev_action)
    else:
        assert results["encoded_obs"].size() == (
            batch,
            obs_len,
            sldgu.hparams.hidden_dim,
        )
        assert results["encoded_prev_action"].size() == (
            batch,
            prev_action_len,
            sldgu.hparams.hidden_dim,
        )
