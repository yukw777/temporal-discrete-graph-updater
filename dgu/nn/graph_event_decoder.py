import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Optional

from dgu.nn.utils import masked_mean
from dgu.constants import EVENT_TYPES


class EventTypeHead(nn.Module):
    def __init__(self, graph_event_embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(graph_event_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(EVENT_TYPES)),
        )
        self.autoregressive_linear = nn.Sequential(
            nn.Linear(len(EVENT_TYPES), graph_event_embedding_dim),
            nn.ReLU(),
        )

    def forward(self, graph_event_embeddings: torch.Tensor) -> torch.Tensor:
        """
        graph_event_embeddings: (batch, graph_event_embedding_dim)

        output: event_type_logits: (batch, num_event_type)
        """
        # logits
        return self.linear(graph_event_embeddings)
        # (batch, num_event_type)

    def get_autoregressive_embedding(
        self, graph_event_embeddings: torch.Tensor, event_type_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        graph_event_embeddings: (batch, graph_event_embedding_dim)
        event_type_ids: (batch)

        output: autoregressive_embedding, (batch, graph_event_embedding_dim)
        """
        # autoregressive embedding
        # get the one hot encoding of event
        one_hot_event_type = F.one_hot(
            event_type_ids, num_classes=len(EVENT_TYPES)
        ).float()
        # (batch, num_event_type)
        # pass it through a linear layer
        encoded_event_type = self.autoregressive_linear(one_hot_event_type)
        # (batch, graph_event_embedding_dim)
        # add it to graph_event_embeddings to calculate the autoregressive embedding
        return graph_event_embeddings + encoded_event_type
        # (batch, graph_event_embedding_dim)


class EventNodeHead(nn.Module):
    def __init__(
        self,
        node_embedding_dim: int,
        autoregressive_embedding_dim: int,
        hidden_dim: int,
        key_query_dim: int,
    ) -> None:
        super().__init__()
        self.key_query_dim = key_query_dim
        self.key_linear = nn.Sequential(
            nn.Linear(node_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, key_query_dim),
        )
        self.query_linear = nn.Sequential(
            nn.Linear(autoregressive_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, key_query_dim),
        )
        self.autoregressive_linear = nn.Linear(
            key_query_dim, autoregressive_embedding_dim
        )

    def forward(
        self, autoregressive_embedding: torch.Tensor, node_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        autoregressive_embedding: (batch, autoregressive_embedding_dim)
        node_embeddings: (batch, num_node, node_embedding_dim)

        output:
            node_logits: (batch, num_node)
            key: (batch, num_node, key_query_dim)
        """
        if node_embeddings.size(1) == 0:
            # if there are no nodes, just return
            batch = node_embeddings.size(0)
            return torch.empty(batch, 0, device=node_embeddings.device), torch.empty(
                batch, 0, self.key_query_dim
            )

        # calculate the key from node_embeddings
        key = self.key_linear(node_embeddings)
        # (batch, num_node, key_query_dim)

        # calculate the query from autoregressive_embedding
        query = self.query_linear(autoregressive_embedding)
        # (batch, key_query_dim)

        return torch.matmul(key, query.unsqueeze(-1)).squeeze(-1), key
        # node_logits: (batch, num_node)
        # key: (batch, num_node, key_query_dim)

    def update_autoregressive_embedding(
        self,
        autoregressive_embedding: torch.Tensor,
        node_ids: torch.Tensor,
        node_embeddings: torch.Tensor,
        key: torch.Tensor,
    ) -> torch.Tensor:
        """
        autoregressive_embedding: (batch, autoregressive_embedding_dim)
        node_ids: (batch)
        node_embeddings: (batch, num_node, node_embedding_dim)
        key: (batch, num_node, key_query_dim)

        output: updated autoregressive embedding
            (batch, autoregressive_embedding_dim)
        """
        if node_embeddings.size(1) == 0:
            # if there are no nodes, just return without updating
            return autoregressive_embedding
        # get the one hot encoding of the selected nodes
        one_hot_selected_node = F.one_hot(
            node_ids, num_classes=node_embeddings.size(1)
        ).float()
        # (batch, num_node)
        # multiply by the key
        selected_node_embeddings = torch.bmm(
            one_hot_selected_node.unsqueeze(1), key
        ).squeeze(1)
        # (batch, key_query_dim)
        # pass it through a linear layer
        selected_node_embeddings = self.autoregressive_linear(selected_node_embeddings)
        # (batch, hidden_dim)
        # add it to the autoregressive embedding
        # NOTE: make sure not to do an in-place += here as it messes with gradients
        return autoregressive_embedding + selected_node_embeddings
        # (batch, hidden_dim)


class EventStaticLabelHead(nn.Module):
    def __init__(
        self,
        autoregressive_embedding_dim: int,
        label_embedding_dim: int,
        hidden_dim: int,
        key_query_dim: int,
    ) -> None:
        super().__init__()
        self.label_embedding_dim = label_embedding_dim

        self.key_linear = nn.Sequential(
            nn.Linear(self.label_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, key_query_dim),
        )
        self.query_linear = nn.Sequential(
            nn.Linear(autoregressive_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, key_query_dim),
        )

    def forward(
        self, autoregressive_embedding: torch.Tensor, label_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        autoregressive_embedding: (batch, autoregressive_embedding_dim)
        label_embeddings: (num_label, label_embedding_dim)

        output:
            label_logits: (batch, num_label), logits for nodes first, then edges
        """
        # calculate the key from label_embeddings
        key = self.key_linear(label_embeddings)
        # (num_label, key_query_dim)

        # calculate the query from event_type and autoregressive_embedding
        query = self.query_linear(autoregressive_embedding)
        # (batch, key_query_dim)

        # multiply key and query to calculate the logits
        return torch.matmul(query, key.transpose(0, 1))


class RNNGraphEventDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        aggr_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim + 4 * aggr_dim, hidden_dim)
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(
        self,
        input_event_embedding: torch.Tensor,
        aggr_obs_graph: torch.Tensor,
        obs_mask: torch.Tensor,
        aggr_prev_action_graph: torch.Tensor,
        prev_action_mask: torch.Tensor,
        aggr_graph_obs: torch.Tensor,
        aggr_graph_prev_action: torch.Tensor,
        node_mask: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        input:
            input_event_embedding: (batch, input_dim)
            aggr_obs_graph: (batch, obs_len, aggr_dim)
            obs_mask: (batch, obs_len)
            aggr_prev_action_graph: (batch, prev_action_len, aggr_dim)
            prev_action_mask: (batch, prev_action_len)
            aggr_graph_obs: (batch, num_node, aggr_dim)
            aggr_graph_prev_action: (batch, num_node, aggr_dim)
            node_mask: (batch, num_node)
            hidden: (batch, hidden_dim)

        output: (batch, hidden_dim)
        """
        mean_aggr_obs_graph = masked_mean(aggr_obs_graph, obs_mask)
        # (batch, aggr_dim)
        mean_aggr_graph_obs = masked_mean(aggr_graph_obs, node_mask)
        # (batch, aggr_dim)
        mean_aggr_prev_action_graph = masked_mean(
            aggr_prev_action_graph, prev_action_mask
        )
        # (batch, aggr_dim)
        mean_aggr_graph_prev_action = masked_mean(aggr_graph_prev_action, node_mask)
        # (batch, aggr_dim)
        gru_input = self.linear(
            torch.cat(
                [
                    input_event_embedding,
                    mean_aggr_obs_graph,
                    mean_aggr_graph_obs,
                    mean_aggr_prev_action_graph,
                    mean_aggr_graph_prev_action,
                ],
                dim=1,
            )
        )
        # (batch, hidden_dim)
        return self.gru_cell(gru_input, hidden)
        # (batch, hidden_dim)
