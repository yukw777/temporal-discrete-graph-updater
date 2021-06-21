import torch
import torch.nn as nn

from typing import Dict, Tuple
from torch_geometric.nn.models.tgn import TimeEncoder


class TemporalGraphNetwork(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # memory
        self.memory: Dict[int, torch.Tensor] = {}

        # last updated timestamp
        self.last_update: Dict[int, int] = {}

        # time encoder
        self.time_encoder = TimeEncoder(hidden_dim)

    def message(
        self,
        event_type_ids: torch.Tensor,
        src_ids: torch.Tensor,
        src_mask: torch.Tensor,
        dst_ids: torch.Tensor,
        dst_mask: torch.Tensor,
        event_embeddings: torch.Tensor,
        event_timestamps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate graph event messages. We concatenate the event type,
        source and destination memories, time embedding and event embedding.

        event_type_ids: (event_seq_len)
        src_ids: (event_seq_len)
        src_mask: (event_seq_len)
        dst_ids: (event_seq_len)
        dst_mask: (event_seq_len)
        event_embeddings: (event_seq_len, hidden_dim)
        event_timestamps: (event_seq_len)

        output:
            src_messages: (event_seq_len, 5 * hidden_dim)
            dst_messages: (event_seq_len, 5 * hidden_dim)
        """
        # repeat event type id for event type embeddings
        event_type_embs = event_type_ids.unsqueeze(-1).expand(-1, self.hidden_dim)
        # (event_seq_len, hidden_dim)

        # use the memory for node embeddings
        src_embs = torch.stack([self.memory[src_id] for src_id in src_ids.tolist()]).to(
            src_ids.device
        )
        src_embs *= src_mask.unsqueeze(-1)
        # (event_seq_len, hidden_dim)
        dst_embs = torch.stack([self.memory[dst_id] for dst_id in dst_ids.tolist()]).to(
            dst_ids.device
        )
        dst_embs *= dst_mask.unsqueeze(-1)
        # (event_seq_len, hidden_dim)

        # fetch last update timestamps
        src_last_update = torch.tensor(
            [self.last_update[src_id] for src_id in src_ids.tolist()]
        ).to(src_mask.device)
        # (event_seq_len)
        dst_last_update = torch.tensor(
            [self.last_update[dst_id] for dst_id in dst_ids.tolist()]
        ).to(dst_mask.device)
        # (event_seq_len)

        # multiply src_last_update and dst_last_update by dst_mask so that we
        # only subtract last update timestamps for edge events
        # then pass them through the time encoder to get the embeddings
        src_timestamp_emb = self.time_encoder(
            event_timestamps - src_last_update * dst_mask
        )
        # (event_seq_len, hidden_dim)
        dst_timestamp_emb = self.time_encoder(
            event_timestamps - dst_last_update * dst_mask
        )
        # (event_seq_len, hidden_dim)

        src_messages = torch.cat(
            [
                event_type_embs,
                src_embs,
                dst_embs,
                src_timestamp_emb,
                event_embeddings,
            ],
            dim=1,
        )
        # (event_seq_len, 5 * hidden_dim)
        dst_messages = torch.cat(
            [
                event_type_embs,
                dst_embs,
                src_embs,
                dst_timestamp_emb,
                event_embeddings,
            ],
            dim=1,
        )
        # (event_seq_len, 5 * hidden_dim)
        return src_messages, dst_messages
