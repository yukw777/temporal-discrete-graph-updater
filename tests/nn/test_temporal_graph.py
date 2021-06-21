import pytest
import random
import torch

from dgu.nn.temporal_graph import TemporalGraphNetwork
from dgu.constants import EVENT_TYPES


@pytest.mark.parametrize(
    "hidden_dim,event_seq_len,num_nodes,max_timestamp",
    [
        (12, 5, 4, 10),
        (24, 12, 10, 15),
    ],
)
def test_tgn_message(hidden_dim, event_seq_len, num_nodes, max_timestamp):
    tgn = TemporalGraphNetwork(hidden_dim)
    for i in range(num_nodes):
        tgn.memory[i] = torch.rand(hidden_dim)
        tgn.last_update[i] = random.randint(0, max_timestamp)
    src_messages, dst_messages = tgn.message(
        torch.randint(len(EVENT_TYPES), (event_seq_len,)),
        torch.randint(num_nodes, (event_seq_len,)),
        torch.randint(2, (event_seq_len,)).float(),
        torch.randint(num_nodes, (event_seq_len,)),
        torch.randint(2, (event_seq_len,)).float(),
        torch.rand(event_seq_len, hidden_dim),
        torch.randint(5, (event_seq_len,)),
    )
    assert src_messages.size() == (event_seq_len, 5 * hidden_dim)
    assert dst_messages.size() == (event_seq_len, 5 * hidden_dim)
