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


class EventNodeHead(nn.Module):
    def __init__(self, hidden_dim: int, key_query_dim: int) -> None:
        super().__init__()
        self.key_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, key_query_dim),
        )
        self.query_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, key_query_dim),
        )
        self.autoregressive_linear = nn.Linear(key_query_dim, hidden_dim)

    def forward(
        self, autoregressive_embedding: torch.Tensor, node_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        autoregressive_embedding: (batch, hidden_dim)
        node_embeddings: (num_node, hidden_dim)

        output:
            node_logits: (batch, num_node)
            autoregressive_embedding: (batch, hidden_dim)
        """
        # calculate the key from node_embeddings
        key = self.key_linear(node_embeddings)
        # (num_node, key_query_dim)

        # calculate the query from autoregressive_embedding
        query = self.query_linear(autoregressive_embedding)
        # (batch, key_query_dim)

        node_logits = torch.matmul(query, key.transpose(0, 1))
        # (batch, num_node)

        # autoregressive embedding
        # get the one hot encoding of the selected nodes
        one_hot_selected_node = (
            F.one_hot(node_logits.argmax(dim=1), num_classes=node_embeddings.size(0))
            .float()
            .to(autoregressive_embedding.device)
        )
        # (batch, num_node)
        # multiply by the key
        selected_node_embeddings = torch.matmul(one_hot_selected_node, key)
        # (batch, key_query_dim)
        # pass it through a linear layer
        selected_node_embeddings = self.autoregressive_linear(selected_node_embeddings)
        # (batch, hidden_dim)
        # add it to the autoregressive embedding
        autoregressive_embedding += selected_node_embeddings
        # (batch, hidden_dim)

        return node_logits, autoregressive_embedding
