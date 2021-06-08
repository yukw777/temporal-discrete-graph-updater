import pytest
import torch

from dgu.nn.graph_event_decoder import EventTypeHead
from dgu.constants import EVENT_TYPES


@pytest.mark.parametrize("hidden_dim,batch", [(12, 1), (64, 8)])
def test_event_type_head(hidden_dim, batch):
    head = EventTypeHead(hidden_dim)
    logits, autoregressive_embedding = head(torch.rand(batch, hidden_dim))
    assert logits.size() == (batch, len(EVENT_TYPES))
    assert autoregressive_embedding.size() == (batch, hidden_dim)
