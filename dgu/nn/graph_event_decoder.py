import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Dict, Optional

from dgu.constants import EVENT_TYPES, EVENT_TYPE_ID_MAP


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

    def forward(
        self, graph_event_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        graph_event_embeddings: (batch, graph_event_embedding_dim)

        output:
            event_type_logits: (batch, num_event_type)
            autoregressive_embedding: (batch, graph_event_embedding_dim)
        """
        # logits
        event_type_logits = self.linear(graph_event_embeddings)
        # (batch, num_event_type)

        # autoregressive embedding
        # get the one hot encoding of event
        one_hot_event_type = F.one_hot(
            event_type_logits.argmax(dim=1), num_classes=len(EVENT_TYPES)
        ).float()
        # (batch, num_event_type)
        # pass it through a linear layer
        encoded_event_type = self.autoregressive_linear(one_hot_event_type)
        # (batch, hidden_dim)
        # add it to graph_event_embeddings to calculate the autoregressive embedding
        autoregressive_embedding = graph_event_embeddings + encoded_event_type
        # (batch, hidden_dim)

        return event_type_logits, autoregressive_embedding


class EventNodeHead(nn.Module):
    def __init__(
        self,
        node_embedding_dim: int,
        autoregressive_embedding_dim: int,
        hidden_dim: int,
        key_query_dim: int,
    ) -> None:
        super().__init__()
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
            autoregressive_embedding: (batch, autoregressive_embedding_dim)
        """
        if node_embeddings.size(1) == 0:
            # if there are no nodes, just return
            return (
                torch.zeros(node_embeddings.size(0), 0, device=node_embeddings.device),
                autoregressive_embedding,
            )

        # calculate the key from node_embeddings
        key = self.key_linear(node_embeddings)
        # (batch, num_node, key_query_dim)

        # calculate the query from autoregressive_embedding
        query = self.query_linear(autoregressive_embedding)
        # (batch, key_query_dim)

        node_logits = torch.matmul(key, query.unsqueeze(-1)).squeeze(-1)
        # (batch, num_node)

        # autoregressive embedding
        # get the one hot encoding of the selected nodes
        one_hot_selected_node = F.one_hot(
            node_logits.argmax(dim=1), num_classes=node_embeddings.size(1)
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
        updated_autoregressive_embedding = (
            autoregressive_embedding + selected_node_embeddings
        )
        # (batch, hidden_dim)

        return node_logits, updated_autoregressive_embedding


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


class StaticLabelGraphEventDecoder(nn.Module):
    def __init__(
        self,
        graph_event_embedding_dim: int,
        node_embedding_dim: int,
        label_embedding_dim: int,
        hidden_dim: int,
        key_query_dim: int,
    ) -> None:
        super().__init__()
        self.event_type_head = EventTypeHead(graph_event_embedding_dim, hidden_dim)
        self.event_src_head = EventNodeHead(
            node_embedding_dim, graph_event_embedding_dim, hidden_dim, key_query_dim
        )
        self.event_dst_head = EventNodeHead(
            node_embedding_dim, graph_event_embedding_dim, hidden_dim, key_query_dim
        )
        self.event_label_head = EventStaticLabelHead(
            graph_event_embedding_dim, label_embedding_dim, hidden_dim, key_query_dim
        )

    def forward(
        self,
        graph_event_embeddings: torch.Tensor,
        node_embeddings: torch.Tensor,
        label_embeddings: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Based on the graph event embeddings and node embeddings, calculate
        event type logits, source node logits, destination node logits and
        label logits for new graph events.

        graph_event_embeddings: (batch, graph_event_embedding_dim)
        node_embeddings: (batch, num_node, node_embedding_dim)
        label_embeddings: (num_label, label_embedding_dim)

        output: {
            "event_type_logits": (batch, num_event_type),
            "src_logits": (batch, num_node),
            "dst_logits": (batch, num_node),
            "label_logits": (batch, num_label),
        }
        """
        event_type_logits, autoregressive_embedding = self.event_type_head(
            graph_event_embeddings
        )
        src_logits, autoregressive_embedding = self.event_src_head(
            autoregressive_embedding, node_embeddings
        )
        dst_logits, autoregressive_embedding = self.event_dst_head(
            autoregressive_embedding, node_embeddings
        )
        label_logits = self.event_label_head(autoregressive_embedding, label_embeddings)
        return {
            "event_type_logits": event_type_logits,
            "src_logits": src_logits,
            "dst_logits": dst_logits,
            "label_logits": label_logits,
        }


class RNNGraphEventDecoder(nn.Module):
    def __init__(
        self,
        event_type_embedding_dim: int,
        node_embedding_dim: int,
        label_embedding_dim: int,
        delta_g_dim: int,
        hidden_dim: int,
        graph_event_decoder: StaticLabelGraphEventDecoder,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(
            event_type_embedding_dim
            + 2 * node_embedding_dim
            + label_embedding_dim
            + delta_g_dim,
            hidden_dim,
        )
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.graph_event_decoder = graph_event_decoder
        self.event_type_embeddings = nn.Embedding(
            len(EVENT_TYPES),
            event_type_embedding_dim,
            padding_idx=EVENT_TYPE_ID_MAP["pad"],
        )

    def forward(
        self,
        event_type_ids: torch.Tensor,
        event_src_ids: torch.Tensor,
        event_dst_ids: torch.Tensor,
        event_label_ids: torch.Tensor,
        delta_g: torch.Tensor,
        node_embeddings: torch.Tensor,
        label_embeddings: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        input:
            event_type_ids: (batch)
            event_src_ids: (batch)
            event_dst_ids: (batch)
            event_label_ids: (batch)
            delta_g: (batch, delta_g_dim)
            node_embeddings: (batch, num_node, node_embedding_dim)
            label_embeddings: (num_label, label_embedding_dim)
            hidden: (batch, hidden_dim)

        output:
            {
                event_type_logits: (batch, num_event_type)
                src_logits: (batch, num_node)
                dst_logits: (batch, num_node)
                label_logits: (batch, num_label)
                new_hidden: (batch, hidden_dim)
            }
        """
        gru_input = self.linear(
            torch.cat(
                [
                    self.get_decoder_input(
                        event_type_ids,
                        event_src_ids,
                        event_dst_ids,
                        event_label_ids,
                        node_embeddings,
                        label_embeddings,
                    ),
                    delta_g,
                ],
                dim=1,
            )
        )
        new_hidden = self.gru_cell(gru_input, hidden)
        results = self.graph_event_decoder(
            new_hidden, node_embeddings, label_embeddings
        )
        results["new_hidden"] = new_hidden
        return results

    def get_decoder_input(
        self,
        event_type_ids: torch.Tensor,
        event_src_ids: torch.Tensor,
        event_dst_ids: torch.Tensor,
        event_label_ids: torch.Tensor,
        node_embeddings: torch.Tensor,
        label_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Turn the graph events into decoder inputs by concatenating their embeddings.

        input:
            event_type_ids: (batch)
            event_src_ids: (batch)
            event_dst_ids: (batch)
            event_label_ids: (batch)
            node_embeddings: (batch, num_node, node_embedding_dim)
            label_embeddings: (num_label, label_embedding_dim)

        output: (batch, decoder_input_dim)
            where decoder_input_dim =
                event_type_embedding_dim +
                2 * node_embedding_dim +
                label_embedding_dim
        """
        batch, num_node, node_embedding_dim = node_embeddings.size()

        # event type embeddings
        event_type_embs = self.event_type_embeddings(event_type_ids)
        # (batch, event_type_embedding_dim)

        # event label embeddings
        event_label_embs = torch.zeros(
            batch, label_embeddings.size(1), device=event_type_ids.device
        )
        # (batch, label_embedding_dim)
        node_add_mask = event_type_ids == EVENT_TYPE_ID_MAP["node-add"]
        # (batch)
        node_delete_mask = event_type_ids == EVENT_TYPE_ID_MAP["node-delete"]
        # (batch)
        edge_event_mask = torch.logical_or(
            event_type_ids == EVENT_TYPE_ID_MAP["edge-add"],
            event_type_ids == EVENT_TYPE_ID_MAP["edge-delete"],
        )
        # (batch)
        label_mask = node_add_mask.logical_or(node_delete_mask).logical_or(
            edge_event_mask
        )
        # (batch)
        event_label_embs[label_mask] = label_embeddings[event_label_ids[label_mask]]
        # (batch, label_embedding_dim)

        # event source/destination node embeddings
        event_src_embs = torch.zeros(
            batch, node_embedding_dim, device=event_type_ids.device
        )
        # (batch, node_embedding_dim)
        event_dst_embs = torch.zeros(
            batch, node_embedding_dim, device=event_type_ids.device
        )
        # (batch, node_embedding_dim)
        src_node_mask = node_delete_mask.logical_or(edge_event_mask)
        if src_node_mask.any():
            # (batch)
            event_src_embs[src_node_mask] = node_embeddings[
                F.one_hot(event_src_ids, num_classes=num_node).bool()
            ][src_node_mask]
            # (batch, node_embedding_dim)

        if edge_event_mask.any():
            event_dst_embs[edge_event_mask] = node_embeddings[
                F.one_hot(event_dst_ids, num_classes=num_node).bool()
            ][edge_event_mask]
            # (batch, node_embedding_dim)

        return torch.cat(
            [event_type_embs, event_src_embs, event_dst_embs, event_label_embs], dim=1
        )
        # (batch, decoder_input_dim)
        # where decoder_input_dim =
        #       event_type_embedding_dim +
        #       2 * node_embedding_dim +
        #       label_embedding_dim
