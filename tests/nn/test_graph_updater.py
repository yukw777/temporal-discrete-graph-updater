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
    "prev_num_node,batch,obs_len,prev_action_len",
    [(0, 1, 10, 5), (0, 8, 20, 10), (10, 8, 20, 10)],
)
def test_sldgu_f_delta(sldgu, prev_num_node, batch, obs_len, prev_action_len):
    delta_g = sldgu.f_delta(
        torch.rand(prev_num_node, sldgu.hparams.hidden_dim),
        torch.rand(batch, obs_len, sldgu.hparams.hidden_dim),
        torch.randint(2, (batch, obs_len)).float(),
        torch.rand(batch, prev_action_len, sldgu.hparams.hidden_dim),
        torch.randint(2, (batch, prev_action_len)).float(),
    )
    assert delta_g.size() == (batch, 4 * sldgu.hparams.hidden_dim)
    assert delta_g.isnan().sum() == 0


@pytest.mark.parametrize(
    "batch,obs_len,prev_action_len,prev_num_node,prev_num_edge,graph_event_seq_len,"
    "num_node,subgraph_num_node",
    [
        (1, 10, 5, 0, 0, 7, 20, 5),
        (4, 12, 8, 6, 12, 10, 25, 12),
    ],
)
def test_sldgu_forward_training(
    sldgu,
    batch,
    obs_len,
    prev_action_len,
    prev_num_node,
    prev_num_edge,
    graph_event_seq_len,
    num_node,
    subgraph_num_node,
):
    sldgu.train()
    num_label = (
        sldgu.seq2seq.graph_event_decoder.event_label_head.num_node_label
        + sldgu.seq2seq.graph_event_decoder.event_label_head.num_edge_label
    )
    results = sldgu(
        torch.randint(13, (batch, obs_len)),
        torch.randint(2, (batch, obs_len)).float(),
        torch.randint(13, (batch, prev_action_len)),
        torch.randint(2, (batch, prev_action_len)).float(),
        torch.randint(len(EVENT_TYPES), (graph_event_seq_len,)),
        torch.randint(prev_num_node, (graph_event_seq_len,))
        if prev_num_node > 0
        else torch.zeros((graph_event_seq_len,)).long(),
        torch.randint(2, (graph_event_seq_len,)).float(),
        torch.randint(prev_num_node, (graph_event_seq_len,))
        if prev_num_node > 0
        else torch.zeros((graph_event_seq_len,)).long(),
        torch.randint(2, (graph_event_seq_len,)).float(),
        torch.randint(prev_num_edge, (graph_event_seq_len,))
        if prev_num_edge > 0
        else torch.zeros((graph_event_seq_len,)).long(),
        torch.randint(num_label, (graph_event_seq_len,)),
        torch.randint(2, (graph_event_seq_len,)).float(),
        torch.randint(10, (graph_event_seq_len,)).float(),
        torch.randint(prev_num_node, (prev_num_node,))
        if prev_num_node > 0
        else torch.zeros((prev_num_node,)).long(),
        torch.randint(prev_num_edge, (prev_num_edge,))
        if prev_num_edge > 0
        else torch.zeros((prev_num_edge,)).long(),
        torch.randint(prev_num_node, (2, prev_num_edge))
        if prev_num_node > 0
        else torch.zeros((prev_num_edge,)).long(),
        torch.randint(10, (prev_num_edge,)),
        subgraph_node_ids=torch.randint(num_node, (subgraph_num_node,)),
        tgt_event_type_ids=torch.randint(len(EVENT_TYPES), (graph_event_seq_len,)),
        tgt_event_src_ids=torch.randint(prev_num_node, (graph_event_seq_len,))
        if prev_num_node > 0
        else torch.zeros((graph_event_seq_len,)).long(),
        tgt_event_src_mask=torch.randint(2, (graph_event_seq_len,)).float(),
        tgt_event_dst_ids=torch.randint(prev_num_node, (graph_event_seq_len,))
        if prev_num_node > 0
        else torch.zeros((graph_event_seq_len,)).long(),
        tgt_event_dst_mask=torch.randint(2, (graph_event_seq_len,)).float(),
        tgt_event_label_ids=torch.randint(num_label, (graph_event_seq_len,)),
        tgt_event_mask=torch.randint(2, (graph_event_seq_len,)).float(),
    )
    assert results["event_type_logits"].size() == (
        graph_event_seq_len,
        len(EVENT_TYPES),
    )
    assert results["src_logits"].size() == (graph_event_seq_len, subgraph_num_node)
    assert results["dst_logits"].size() == (graph_event_seq_len, subgraph_num_node)
    assert results["label_logits"].size() == (graph_event_seq_len, num_label)


@pytest.mark.parametrize(
    "batch,obs_len,prev_action_len,prev_num_node,prev_num_edge,graph_event_seq_len",
    [
        (1, 10, 5, 0, 0, 7),
        (4, 12, 8, 6, 12, 10),
    ],
)
def test_sldgu_forward_eval(
    sldgu,
    batch,
    obs_len,
    prev_action_len,
    prev_num_node,
    prev_num_edge,
    graph_event_seq_len,
):
    sldgu.eval()
    num_label = (
        sldgu.seq2seq.graph_event_decoder.event_label_head.num_node_label
        + sldgu.seq2seq.graph_event_decoder.event_label_head.num_edge_label
    )
    results = sldgu(
        torch.randint(13, (batch, obs_len)),
        torch.randint(2, (batch, obs_len)).float(),
        torch.randint(13, (batch, prev_action_len)),
        torch.randint(2, (batch, prev_action_len)).float(),
        torch.randint(len(EVENT_TYPES), (graph_event_seq_len,)),
        torch.randint(prev_num_node, (graph_event_seq_len,))
        if prev_num_node > 0
        else torch.zeros((graph_event_seq_len,)).long(),
        torch.randint(2, (graph_event_seq_len,)).float(),
        torch.randint(prev_num_node, (graph_event_seq_len,))
        if prev_num_node > 0
        else torch.zeros((graph_event_seq_len,)).long(),
        torch.randint(2, (graph_event_seq_len,)).float(),
        torch.randint(prev_num_edge, (graph_event_seq_len,))
        if prev_num_edge > 0
        else torch.zeros((graph_event_seq_len,)).long(),
        torch.randint(num_label, (graph_event_seq_len,)),
        torch.randint(2, (graph_event_seq_len,)).float(),
        torch.randint(10, (graph_event_seq_len,)).float(),
        torch.randint(prev_num_node, (prev_num_node,))
        if prev_num_node > 0
        else torch.zeros((prev_num_node,)).long(),
        torch.randint(prev_num_edge, (prev_num_edge,))
        if prev_num_edge > 0
        else torch.zeros((prev_num_edge,)).long(),
        torch.randint(prev_num_node, (2, prev_num_edge))
        if prev_num_node > 0
        else torch.zeros((prev_num_edge,)).long(),
        torch.randint(10, (prev_num_edge,)),
    )
    assert results["decoded_event_type_ids"].dtype == torch.long
    (decoded_len,) = results["decoded_event_type_ids"].size()
    assert decoded_len <= sldgu.hparams.max_decode_len + 1
    assert results["decoded_src_ids"].size() == (decoded_len,)
    assert results["decoded_dst_ids"].size() == (decoded_len,)
    assert results["decoded_label_ids"].size() == (decoded_len,)
