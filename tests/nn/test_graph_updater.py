import pytest
import torch
import torch.nn as nn

from dgu.nn.graph_updater import StaticLabelDiscreteGraphUpdater
from dgu.constants import EVENT_TYPES


@pytest.fixture
def sldgu():
    return StaticLabelDiscreteGraphUpdater(
        64,
        100,
        200,
        100,
        1,
        1,
        3,
        2,
        128,
        100,
        nn.Embedding(200, 100),
        torch.rand(20, 100),
        torch.rand(10, 100),
    )


@pytest.mark.parametrize("batch,seq_len", [(1, 10), (8, 24)])
def test_sldgu_encode_text(sldgu, batch, seq_len):
    assert sldgu.encode_text(
        torch.randint(200, (batch, seq_len)), torch.randint(2, (batch, seq_len)).float()
    ).size() == (batch, seq_len, sldgu.hidden_dim)


@pytest.mark.parametrize(
    "prev_num_node,batch,obs_len,prev_action_len",
    [(0, 1, 10, 5), (10, 8, 20, 10)],
)
def test_sldgu_f_delta(sldgu, prev_num_node, batch, obs_len, prev_action_len):
    assert (
        sldgu.f_delta(
            torch.rand(prev_num_node, sldgu.hidden_dim),
            torch.rand(batch, obs_len, sldgu.hidden_dim),
            torch.randint(2, (batch, obs_len)).float(),
            torch.rand(batch, prev_action_len, sldgu.hidden_dim),
            torch.randint(2, (batch, prev_action_len)).float(),
        ).size()
        == (batch, 4 * sldgu.hidden_dim)
    )


@pytest.mark.parametrize(
    "batch,obs_len,prev_action_len,prev_num_node,graph_event_seq_len",
    [
        (1, 10, 5, 0, 7),
        (4, 12, 8, 6, 10),
    ],
)
def test_sldgu_forward_training(
    sldgu, batch, obs_len, prev_action_len, prev_num_node, graph_event_seq_len
):
    sldgu.train()
    num_label = (
        sldgu.seq2seq.graph_event_decoder.event_label_head.num_node_label
        + sldgu.seq2seq.graph_event_decoder.event_label_head.num_edge_label
    )
    results = sldgu(
        torch.randint(200, (batch, obs_len)),
        torch.randint(2, (batch, obs_len)).float(),
        torch.randint(200, (batch, prev_action_len)),
        torch.randint(2, (batch, prev_action_len)).float(),
        torch.rand(prev_num_node, sldgu.hidden_dim),
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
    assert results["src_logits"].size() == (graph_event_seq_len, prev_num_node)
    assert results["dst_logits"].size() == (graph_event_seq_len, prev_num_node)
    assert results["label_logits"].size() == (graph_event_seq_len, num_label)
