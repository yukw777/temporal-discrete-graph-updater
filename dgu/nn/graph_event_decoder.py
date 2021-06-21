import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Dict

from dgu.constants import EVENT_TYPES, EVENT_TYPE_ID_MAP
from dgu.nn.utils import compute_masks_from_event_type_ids


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
        self.event_src_head = EventNodeHead(hidden_dim, key_query_dim)
        self.event_dst_head = EventNodeHead(hidden_dim, key_query_dim)
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
        label_logits = self.event_label_head(
            autoregressive_embedding, event_type_logits.argmax(dim=1)
        )
        return {
            "event_type_logits": event_type_logits,
            "src_logits": src_logits,
            "dst_logits": dst_logits,
            "label_logits": label_logits,
        }


class StaticLabelGraphEventEncoder(nn.Module):
    def forward(
        self,
        event_type_id: torch.Tensor,
        src_id: torch.Tensor,
        src_mask: torch.Tensor,
        dst_id: torch.Tensor,
        dst_mask: torch.Tensor,
        label_id: torch.Tensor,
        label_mask: torch.Tensor,
        node_embeddings: torch.Tensor,
        label_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode the given batch of graph events. We simply concatenate all
        the components.

        event_type_id: (batch, graph_event_seq_len)
        src_id: (batch, graph_event_seq_len)
        src_mask: (batch, graph_event_seq_len)
        dst_id: (batch, graph_event_seq_len)
        dst_mask: (batch, graph_event_seq_len)
        label_id: (batch, graph_event_seq_len)
        label_mask: (batch, graph_event_seq_len)
        node_embeddings: (num_node, hidden_dim)
        label_embeddings: (num_label, hidden_dim)

        output: (batch, graph_event_seq_len, 4*hidden_dim)
        """
        src_embeddings = node_embeddings[src_id]
        src_embeddings *= src_mask.unsqueeze(-1)
        # (batch, graph_event_seq_len, hidden_dim)

        dst_embeddings = node_embeddings[dst_id]
        dst_embeddings *= dst_mask.unsqueeze(-1)
        # (batch, graph_event_seq_len, hidden_dim)

        label_embeddings = label_embeddings[label_id]
        label_embeddings *= label_mask.unsqueeze(-1)
        # (batch, graph_event_seq_len, hidden_dim)

        return torch.cat(
            [
                event_type_id.unsqueeze(-1).expand(-1, -1, src_embeddings.size(2)),
                src_embeddings,
                dst_embeddings,
                label_embeddings,
            ],
            dim=2,
        )
        # (batch, graph_event_seq_len, 4*hidden_dim)


class RNNGraphEventSeq2Seq(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        max_decode_len: int,
        graph_event_encoder: StaticLabelGraphEventEncoder,
        graph_event_decoder: StaticLabelGraphEventDecoder,
    ) -> None:
        super().__init__()
        self.max_decode_len = max_decode_len
        self.rnn_encoder = nn.GRU(4 * hidden_dim, hidden_dim, batch_first=True)
        self.rnn_decoder = nn.GRU(4 * hidden_dim, hidden_dim, batch_first=True)
        self.graph_event_encoder = graph_event_encoder
        self.graph_event_decoder = graph_event_decoder

    def forward(self, input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        input: {
            delta_g: (batch, obs_seq_len, 4 * hidden_dim)
            tgt_event_mask: (batch, graph_event_seq_len)
                Used for teacher forcing.
            tgt_event_type_ids: (batch, graph_event_seq_len)
                Used for teacher forcing.
            tgt_event_src_ids: (batch, graph_event_seq_len)
                Used for teacher forcing.
            tgt_event_src_mask: (batch, graph_event_seq_len)
                Used for teacher forcing.
            tgt_event_dst_ids: (batch, graph_event_seq_len)
                Used for teacher forcing.
            tgt_event_dst_mask: (batch, graph_event_seq_len)
                Used for teacher forcing.
            tgt_event_label_ids: (batch, graph_event_seq_len)
                Used for teacher forcing.
            node_embeddings: (num_node, hidden_dim)
        }

        output:
        if training:
            {
                event_type_logits: (batch, graph_event_seq_len, num_event_type)
                src_logits: (batch, graph_event_seq_len, num_node)
                dst_logits: (batch, graph_event_seq_len, num_node)
                label_logits: (batch, graph_event_seq_len, num_label)
            }
        else:
            {
                decoded_event_type_ids: (batch, decoded_len),
                decoded_src_ids: (batch, decoded_len),
                decoded_dst_ids: (batch, decoded_len),
                decoded_label_ids: (batch, decoded_len),
            }
        """
        _, context = self.rnn_encoder(input["delta_g"])
        # (1, batch, hidden_dim)
        if self.training:
            # teacher forcing
            encoded_graph_event_seq = self.graph_event_encoder(
                input["tgt_event_type_ids"],
                input["tgt_event_src_ids"],
                input["tgt_event_src_mask"],
                input["tgt_event_dst_ids"],
                input["tgt_event_dst_mask"],
                input["tgt_event_label_ids"],
                input["tgt_event_mask"],
                input["node_embeddings"],
                self.graph_event_decoder.event_label_head.label_embeddings,
            )
            # (batch, graph_event_seq_len, 4 * hidden_dim)
            output, _ = self.rnn_decoder(encoded_graph_event_seq, context)
            # (batch, graph_event_seq_len, hidden_dim)
            batch, graph_event_seq_len, _ = output.size()
            decoded_graph_event_seq_results = self.graph_event_decoder(
                output.flatten(end_dim=1), input["node_embeddings"]
            )
            return {
                "event_type_logits": decoded_graph_event_seq_results[
                    "event_type_logits"
                ].view(batch, graph_event_seq_len, -1),
                "src_logits": decoded_graph_event_seq_results["src_logits"].view(
                    batch, graph_event_seq_len, -1
                ),
                "dst_logits": decoded_graph_event_seq_results["dst_logits"].view(
                    batch, graph_event_seq_len, -1
                ),
                "label_logits": decoded_graph_event_seq_results["label_logits"].view(
                    batch, graph_event_seq_len, -1
                ),
            }
        return self.greedy_decode(context, input["node_embeddings"])

    def greedy_decode(
        self,
        context: torch.Tensor,
        node_embeddings: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Start with "start" event and greedy decode.

        context: (1, batch, hidden_dim)
        node_embeddings: (num_node, hidden_dim)

        output: {
            'decoded_event_type_ids': (batch, decoded_len),
            'decoded_src_ids': (batch, decoded_len),
            'decoded_dst_ids': (batch, decoded_len),
            'decoded_label_ids': (batch, decoded_len),
        }
        """
        # initial start events
        batch = context.size(1)
        decoded_event_type_ids = [
            torch.tensor(
                [EVENT_TYPE_ID_MAP["start"]] * batch, device=context.device
            ).unsqueeze(-1)
            # (batch, 1)
        ]
        # placeholder node IDs
        decoded_src_ids = [
            torch.tensor([0] * batch, device=context.device).unsqueeze(-1)
            # (batch, 1)
        ]
        decoded_dst_ids = [
            torch.tensor([0] * batch, device=context.device).unsqueeze(-1)
            # (batch, 1)
        ]

        # initial pad labels
        decoded_label_ids = [
            torch.tensor([0] * batch, device=context.device).unsqueeze(-1)
            # (batch, 1)
        ]

        # this controls when to stop decoding for each sequence in the batch
        end_event_mask = torch.tensor([False] * batch, device=context.device)
        # (batch)

        prev_hidden = context
        # (1, batch, hidden_dim)
        for _ in range(self.max_decode_len):
            prev_event_type_ids = decoded_event_type_ids[-1]
            # (batch, 1)
            (
                prev_event_mask,
                prev_src_mask,
                prev_dst_mask,
            ) = compute_masks_from_event_type_ids(prev_event_type_ids)
            # event_mask: (batch, 1)
            # src_mask: (batch, 1)
            # dst_mask: (batch, 1)

            encoded_graph_event_seq = self.graph_event_encoder(
                prev_event_type_ids,
                decoded_src_ids[-1],
                prev_src_mask,
                decoded_dst_ids[-1],
                prev_dst_mask,
                decoded_label_ids[-1],
                prev_event_mask,
                node_embeddings,
                self.graph_event_decoder.event_label_head.label_embeddings,
            )
            # (batch, 1, 4 * hidden_dim)
            output, h_n = self.rnn_decoder(encoded_graph_event_seq, prev_hidden)
            # output: (batch, 1, hidden_dim)
            # h_n: (1, batch, hidden_dim)

            # decode the generated graph events
            decoded_graph_event_seq_results = self.graph_event_decoder(
                output.squeeze(1), node_embeddings
            )

            # add decoded results to the lists, masking sequences that have "ended".
            decoded_event_type_ids.append(
                decoded_graph_event_seq_results["event_type_logits"]
                .argmax(dim=-1)
                .masked_fill(end_event_mask, EVENT_TYPE_ID_MAP["pad"])
                .unsqueeze(-1)
                # (batch, 1)
            )
            decoded_src_ids.append(
                decoded_graph_event_seq_results["src_logits"]
                .argmax(dim=-1)
                .masked_fill(end_event_mask, 0)
                .unsqueeze(-1)
                # (batch, 1)
            )
            decoded_dst_ids.append(
                decoded_graph_event_seq_results["dst_logits"]
                .argmax(dim=-1)
                .masked_fill(end_event_mask, 0)
                .unsqueeze(-1)
                # (batch, 1)
            )
            decoded_label_ids.append(
                decoded_graph_event_seq_results["label_logits"]
                .argmax(dim=-1)
                .masked_fill(end_event_mask, 0)
                .unsqueeze(-1)
                # (batch, 1)
            )

            # update end_event_mask
            end_event_mask = end_event_mask.logical_or(
                decoded_event_type_ids[-1].squeeze(1) == EVENT_TYPE_ID_MAP["end"]
            )

            if end_event_mask.all():
                # if all the sequences in the batch have "ended", break
                break

            # update prev_hidden
            prev_hidden = h_n

        return {
            "decoded_event_type_ids": torch.cat(decoded_event_type_ids, dim=-1),
            "decoded_src_ids": torch.cat(decoded_src_ids, dim=-1),
            "decoded_dst_ids": torch.cat(decoded_dst_ids, dim=-1),
            "decoded_label_ids": torch.cat(decoded_label_ids, dim=-1),
        }
