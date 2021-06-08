import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

from dgu.constants import EVENT_TYPES


class EventTypeHead(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(EVENT_TYPES)),
        )
        self.autoregressive_linear = nn.Sequential(
            nn.Linear(len(EVENT_TYPES), hidden_dim),
            nn.ReLU(),
        )

    def forward(
        self, graph_event_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        graph_event_embeddings: (batch, hidden_dim)

        output:
            event_type_logits: (batch, num_event_type)
            autoregressive_embedding: (batch, hidden_dim)
        """
        # logits
        event_type_logits = self.linear(graph_event_embeddings)
        # (batch, num_event_type)

        # autoregressive embedding
        # get the one hot encoding of event
        one_hot_event_type = (
            F.one_hot(event_type_logits.argmax(dim=1), num_classes=len(EVENT_TYPES))
            .float()
            .to(graph_event_embeddings.device)
        )
        # (batch, num_event_type)
        # pass it through a linear layer
        encoded_event_type = self.autoregressive_linear(one_hot_event_type)
        # (batch, hidden_dim)
        # add it to graph_event_embeddings to calculate the autoregressive embedding
        autoregressive_embedding = graph_event_embeddings + encoded_event_type
        # (batch, hidden_dim)

        return event_type_logits, autoregressive_embedding
