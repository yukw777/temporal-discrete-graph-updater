import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Dict

from dgu.constants import EVENT_TYPES, EVENT_TYPE_ID_MAP


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
        if node_embeddings.size(0) == 0:
            # if there are no nodes, just use an empty tensor without taking an argmax
            one_hot_selected_node = torch.empty_like(node_logits)
        else:
            one_hot_selected_node = F.one_hot(
                node_logits.argmax(dim=1), num_classes=node_embeddings.size(0)
            ).float()
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


class EventStaticLabelHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        key_query_dim: int,
        node_label_embeddings: torch.Tensor,
        edge_label_embeddings: torch.Tensor,
    ) -> None:
        super().__init__()
        assert node_label_embeddings.size(1) == edge_label_embeddings.size(1)
        self.label_embedding_dim = node_label_embeddings.size(1)
        self.num_node_label = node_label_embeddings.size(0)
        self.num_edge_label = edge_label_embeddings.size(0)

        self.key_linear = nn.Sequential(
            nn.Linear(self.label_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, key_query_dim),
        )
        self.query_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, key_query_dim),
        )
        self.register_buffer(
            "label_embeddings",
            torch.cat([node_label_embeddings, edge_label_embeddings]),
        )

    def forward(
        self,
        autoregressive_embedding: torch.Tensor,
        event_type: torch.Tensor,
    ) -> torch.Tensor:
        """
        autoregressive_embedding: (batch, hidden_dim)
        event_type: (batch)

        output:
            label_logits: (batch, num_label), logits for nodes first, then edges
        """
        # calculate the key from label_embeddings
        key = self.key_linear(self.label_embeddings)
        # (num_label, key_query_dim)

        # calculate the query from event_type and autoregressive_embedding
        query = self.query_linear(autoregressive_embedding)
        # (batch, key_query_dim)

        # multiply key and query to calculate the logits
        label_logits = torch.matmul(query, key.transpose(0, 1))
        # (batch, num_label)

        # calculate label mask based on event_type
        one_hot_event_type = F.one_hot(event_type, num_classes=len(EVENT_TYPES)).float()
        # (batch, num_event_type)
        mask = torch.hstack(
            [
                # one hot encoding for node add/delete events
                one_hot_event_type[
                    :, EVENT_TYPE_ID_MAP["node-add"] : EVENT_TYPE_ID_MAP["edge-add"]
                ]
                # sum to get one hot encoding for node events
                .sum(1, keepdim=True)
                # repeat for the number of node labels
                .expand(-1, self.num_node_label),
                # one hot encoding for edge add/delete events
                one_hot_event_type[:, EVENT_TYPE_ID_MAP["edge-add"] :]
                # sum to get one hot encoding for edge events
                .sum(1, keepdim=True)
                # repeat for the number of edge labels
                .expand(-1, self.num_edge_label),
            ]
        )
        # (batch, num_label)

        # mask out the label logits and return
        return label_logits * mask


class StaticLabelGraphEventDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        key_query_dim: int,
        node_label_embeddings: torch.Tensor,
        edge_label_embeddings: torch.Tensor,
    ) -> None:
        super().__init__()
        self.event_type_head = EventTypeHead(hidden_dim)
        self.event_src_node_head = EventNodeHead(hidden_dim, key_query_dim)
        self.event_dst_node_head = EventNodeHead(hidden_dim, key_query_dim)
        self.event_label_head = EventStaticLabelHead(
            hidden_dim, key_query_dim, node_label_embeddings, edge_label_embeddings
        )

    def forward(
        self, graph_event_embeddings: torch.Tensor, node_embeddings: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Based on the graph event embeddings and node embeddings, calculate
        event type logits, source node logits, destination node logits and
        label logits for new graph events.

        graph_event_embeddings: (batch, hidden_dim)
        node_embeddings: (num_node, hidden_dim)

        output: {
            "event_type_logits": (batch, num_event_type),
            "src_node_logits": (batch, num_node),
            "dst_node_logits": (batch, num_node),
            "label_logits": (batch, num_label),
        }
        """
        event_type_logits, autoregressive_embedding = self.event_type_head(
            graph_event_embeddings
        )
        src_node_logits, autoregressive_embedding = self.event_src_node_head(
            autoregressive_embedding, node_embeddings
        )
        dst_node_logits, autoregressive_embedding = self.event_dst_node_head(
            autoregressive_embedding, node_embeddings
        )
        label_logits = self.event_label_head(
            autoregressive_embedding, event_type_logits.argmax(dim=1)
        )
        return {
            "event_type_logits": event_type_logits,
            "src_node_logits": src_node_logits,
            "dst_node_logits": dst_node_logits,
            "label_logits": label_logits,
        }


class StaticLabelGraphEventEncoder(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear = nn.Sequential(
            nn.Linear(4 * hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

    def forward(
        self,
        event_type_id: torch.Tensor,
        src_node_id: torch.Tensor,
        src_node_mask: torch.Tensor,
        dst_node_id: torch.Tensor,
        dst_node_mask: torch.Tensor,
        label_id: torch.Tensor,
        node_embeddings: torch.Tensor,
        label_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode the given batch of graph events. We concatenate all the components
        and pass it through two linear layers with a ReLU.

        event_type_id: (batch, graph_event_seq_len)
        src_node_id: (batch, graph_event_seq_len)
        src_node_mask: (batch, graph_event_seq_len)
        dst_node_id: (batch, graph_event_seq_len)
        dst_node_mask: (batch, graph_event_seq_len)
        label_id: (batch, graph_event_seq_len)
        node_embeddings: (num_node, hidden_dim)
        label_embeddings: (num_label, hidden_dim)

        output: (batch, graph_event_seq_len, hidden_dim)
        """
        src_node_embeddings = node_embeddings[src_node_id]
        src_node_embeddings *= src_node_mask.unsqueeze(-1)
        # (batch, graph_event_seq_len, hidden_dim)

        dst_node_embeddings = node_embeddings[src_node_id]
        dst_node_embeddings *= dst_node_mask.unsqueeze(-1)
        # (batch, graph_event_seq_len, hidden_dim)

        label_embeddings = label_embeddings[label_id]
        label_embeddings *= src_node_mask.unsqueeze(-1)
        # (batch, graph_event_seq_len, hidden_dim)

        concat = torch.cat(
            [
                event_type_id.unsqueeze(-1).expand(-1, -1, self.hidden_dim),
                src_node_embeddings,
                dst_node_embeddings,
                label_embeddings,
            ],
            dim=2,
        )
        # (batch, graph_event_seq_len, 4*hidden_dim)
        return self.linear(concat)
        # (batch, graph_event_seq_len, hidden_dim)
