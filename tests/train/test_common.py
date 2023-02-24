import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data.batch import Batch

from tdgu.constants import EVENT_TYPES
from tdgu.data import TWCmdGenGraphEventStepInput
from tdgu.nn.utils import masked_softmax
from tdgu.train.common import TDGULightningModule


@pytest.mark.parametrize("gumbel_tau", [0.3, 0.5])
@pytest.mark.parametrize(
    "max_event_decode_len,max_label_decode_len,gumbel_greedy_decode,batch,obs_len,"
    "prev_action_len,forward_results,prev_batched_graph,expected_decoded_list",
    [
        (
            10,
            2,
            False,
            1,
            3,
            5,
            [
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 1, 0, 0, 0]]
                    ).float(),  # [node-add]
                    "event_src_logits": torch.empty(1, 0),
                    "event_dst_logits": torch.empty(1, 0),
                    "decoded_event_label_word_ids": torch.tensor([[13, 3]]),
                    "decoded_event_label_mask": torch.ones(1, 2).bool(),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 1, 1, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor([[True]]),
                    "encoded_obs": torch.rand(1, 3, 8),
                    "encoded_prev_action": torch.rand(1, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.empty(0).long(),
                        x=torch.empty(0, 0, dtype=torch.long),
                        node_label_mask=torch.empty(0, 0).bool(),
                        node_last_update=torch.empty(0, 2, dtype=torch.long),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2, dtype=torch.long),
                    ),
                    "batch_node_embeddings": torch.empty(1, 0, 8),
                    "batch_node_mask": torch.empty(1, 0).bool(),
                    "self_attn_weights": [torch.rand(1, 1, 1)],
                    "obs_graph_attn_weights": [torch.rand(1, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(1, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(1, 1, 0)],
                    "graph_prev_action_attn_weights": [torch.rand(1, 1, 0)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 1, 0, 0, 0, 0]]
                    ).float(),  # [end]
                    "event_src_logits": torch.rand(1, 1),
                    "event_dst_logits": torch.rand(1, 1),
                    "decoded_event_label_word_ids": torch.tensor([[3]]),
                    "decoded_event_label_mask": torch.tensor([[True]]),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 1, 2, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True, True]]
                    ),
                    "encoded_obs": torch.rand(1, 3, 8),
                    "encoded_prev_action": torch.rand(1, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0]),
                        x=torch.tensor([[13, 3]]),
                        node_label_mask=torch.ones(1, 2).bool(),
                        node_last_update=torch.tensor([[1, 2]]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.ones(1, 1, 8),
                    "batch_node_mask": torch.tensor([[True]]),
                    "self_attn_weights": [torch.rand(1, 1, 2)],
                    "obs_graph_attn_weights": [torch.rand(1, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(1, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(1, 1, 1)],
                    "graph_prev_action_attn_weights": [torch.rand(1, 1, 1)],
                },
            ],
            Batch(
                batch=torch.empty(0, dtype=torch.long),
                x=torch.empty(0, 0, dtype=torch.long),
                node_label_mask=torch.empty(0, 0).bool(),
                node_last_update=torch.empty(0, 2, dtype=torch.long),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=torch.empty(0, 0).long(),
                edge_label_mask=torch.empty(0, 0).bool(),
                edge_last_update=torch.empty(0, 2, dtype=torch.long),
            ),
            [
                {
                    "decoded_event_type_ids": torch.tensor([3]),  # [node-add]
                    "decoded_event_src_ids": torch.zeros(1).long(),
                    "decoded_event_dst_ids": torch.zeros(1).long(),
                    "decoded_event_label_word_ids": torch.tensor([[13, 3]]),  # [player]
                    "decoded_event_label_mask": torch.ones(1, 2).bool(),
                    "updated_batched_graph": Batch(
                        batch=torch.empty(0, dtype=torch.long),
                        x=torch.empty(0, 0, dtype=torch.long),
                        node_label_mask=torch.empty(0, 0).bool(),
                        node_last_update=torch.empty(0, 2, dtype=torch.long),
                        edge_index=torch.empty(2, 0, dtype=torch.long),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2, dtype=torch.long),
                    ),
                    "batch_node_embeddings": torch.empty(1, 0, 8),
                    "batch_node_mask": torch.empty(1, 0).bool(),
                },
                {
                    "decoded_event_type_ids": torch.tensor([2]),  # [end]
                    "decoded_event_src_ids": torch.zeros(1).long(),
                    "decoded_event_dst_ids": torch.zeros(1).long(),
                    "decoded_event_label_word_ids": torch.tensor([[3]]),
                    "decoded_event_label_mask": torch.tensor([[True]]),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0]),
                        x=torch.tensor([[13, 3]]),
                        node_label_mask=torch.ones(1, 2).bool(),
                        node_last_update=torch.tensor([[1, 2]]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.ones(1, 1, 8),
                    "batch_node_mask": torch.tensor([[True]]),
                },
            ],
        ),
        (
            2,
            3,
            False,
            1,
            3,
            5,
            [
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 1, 0, 0, 0]]
                    ).float(),  # [node-add]
                    "event_src_logits": torch.empty(1, 0),
                    "event_dst_logits": torch.empty(1, 0),
                    "decoded_event_label_word_ids": torch.tensor([[13, 3]]),
                    "decoded_event_label_mask": torch.ones(1, 2).bool(),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 1, 1, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor([[True]]),
                    "encoded_obs": torch.rand(1, 3, 8),
                    "encoded_prev_action": torch.rand(1, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.empty(0).long(),
                        x=torch.empty(0, 0, dtype=torch.long),
                        node_label_mask=torch.empty(0, 0).bool(),
                        node_last_update=torch.empty(0, 2).long(),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.empty(1, 0, 8),
                    "batch_node_mask": torch.empty(1, 0).bool(),
                    "self_attn_weights": [torch.rand(1, 1, 1)],
                    "obs_graph_attn_weights": [torch.rand(1, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(1, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(1, 1, 0)],
                    "graph_prev_action_attn_weights": [torch.rand(1, 1, 0)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 1, 0, 0, 0]]
                    ).float(),  # [node-add]
                    "event_src_logits": torch.rand(1, 1),
                    "event_dst_logits": torch.rand(1, 1),
                    "decoded_event_label_word_ids": torch.tensor([[14, 3]]),
                    "decoded_event_label_mask": torch.tensor([[True, True]]),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 1, 2, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True, True]]
                    ),
                    "encoded_obs": torch.rand(1, 3, 8),
                    "encoded_prev_action": torch.rand(1, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0]),
                        x=torch.tensor([[13, 3]]),
                        node_label_mask=torch.ones(1, 2).bool(),
                        node_last_update=torch.tensor([[1, 2]]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.ones(1, 1, 8),
                    "batch_node_mask": torch.tensor([[True]]),
                    "self_attn_weights": [torch.rand(1, 1, 2)],
                    "obs_graph_attn_weights": [torch.rand(1, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(1, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(1, 1, 1)],
                    "graph_prev_action_attn_weights": [torch.rand(1, 1, 1)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 0, 0, 1, 0]]
                    ).float(),  # [edge-add]
                    "event_src_logits": torch.tensor([[0, 1]]).float(),
                    "event_dst_logits": torch.tensor([[1, 0]]).float(),
                    "decoded_event_label_word_ids": torch.tensor([[12, 3]]),
                    "decoded_event_label_mask": torch.tensor([[True, True]]),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 1, 3, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True, True, True]]
                    ),
                    "encoded_obs": torch.rand(1, 3, 8),
                    "encoded_prev_action": torch.rand(1, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0]),
                        x=torch.tensor([[13, 3], [14, 3]]),
                        node_label_mask=torch.ones(2, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [1, 3]]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.zeros(1, 2, 8),
                    "batch_node_mask": torch.tensor([[True, True]]),
                    "self_attn_weights": [torch.rand(1, 1, 3)],
                    "obs_graph_attn_weights": [torch.rand(1, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(1, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(1, 1, 2)],
                    "graph_prev_action_attn_weights": [torch.rand(1, 1, 2)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 1, 0, 0, 0, 0]]
                    ).float(),  # [end]
                    "event_src_logits": torch.rand(1, 2),
                    "event_dst_logits": torch.rand(1, 2),
                    "decoded_event_label_word_ids": torch.tensor([[3]]),
                    "decoded_event_label_mask": torch.tensor([[True]]),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 1, 4, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True, True, True, True]]
                    ),
                    "encoded_obs": torch.rand(1, 3, 8),
                    "encoded_prev_action": torch.rand(1, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0]),
                        x=torch.tensor([[13, 3], [14, 3]]),
                        node_label_mask=torch.ones(2, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [1, 3]]),
                        edge_index=torch.tensor([[1], [0]]),
                        edge_attr=torch.tensor([[12, 3]]),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[1, 4]]),
                    ),
                    "batch_node_embeddings": torch.zeros(1, 2, 8),
                    "batch_node_mask": torch.tensor([[True, True]]),
                    "self_attn_weights": [torch.rand(1, 1, 4)],
                    "obs_graph_attn_weights": [torch.rand(1, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(1, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(1, 1, 2)],
                    "graph_prev_action_attn_weights": [torch.rand(1, 1, 2)],
                },
            ],
            Batch(
                batch=torch.empty(0, dtype=torch.long),
                x=torch.empty(0, 0, dtype=torch.long),
                node_label_mask=torch.empty(0, 0).bool(),
                node_last_update=torch.empty(0, 2).long(),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=torch.empty(0, 0).long(),
                edge_label_mask=torch.empty(0, 0).bool(),
                edge_last_update=torch.empty(0, 2).long(),
            ),
            [
                {
                    "decoded_event_type_ids": torch.tensor([3]),  # [node-add]
                    "decoded_event_src_ids": torch.zeros(1).long(),
                    "decoded_event_dst_ids": torch.zeros(1).long(),
                    "decoded_event_label_word_ids": torch.tensor([[13, 3]]),  # [player]
                    "decoded_event_label_mask": torch.ones(1, 2).bool(),
                    "updated_batched_graph": Batch(
                        batch=torch.empty(0, dtype=torch.long),
                        x=torch.empty(0, 0, dtype=torch.long),
                        node_label_mask=torch.empty(0, 0).bool(),
                        node_last_update=torch.empty(0, 2).long(),
                        edge_index=torch.empty(2, 0, dtype=torch.long),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.empty(1, 0, 8),
                    "batch_node_mask": torch.empty(1, 0).bool(),
                },
                {
                    "decoded_event_type_ids": torch.tensor([3]),  # [node-add]
                    "decoded_event_src_ids": torch.zeros(1).long(),
                    "decoded_event_dst_ids": torch.zeros(1).long(),
                    "decoded_event_label_word_ids": torch.tensor([[14, 3]]),  # [player]
                    "decoded_event_label_mask": torch.ones(1, 2).bool(),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0]),
                        x=torch.tensor([[13, 3]]),
                        node_label_mask=torch.ones(1, 2).bool(),
                        node_last_update=torch.tensor([[1, 2]]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.ones(1, 1, 8),
                    "batch_node_mask": torch.tensor([[True]]),
                },
            ],
        ),
        (
            10,
            3,
            False,
            2,
            3,
            5,
            [
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]
                    ).float(),  # [node-add, node-add]
                    "event_src_logits": torch.empty(2, 0),
                    "event_dst_logits": torch.empty(2, 0),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[13, 3], [13, 3]]
                    ),  # [player, player]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 1, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True], [True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.empty(0).long(),
                        x=torch.empty(0, 0, dtype=torch.long),
                        node_label_mask=torch.empty(0, 0).bool(),
                        node_last_update=torch.empty(0, 2).long(),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.empty(2, 0, 8),
                    "batch_node_mask": torch.empty(2, 0).bool(),
                    "self_attn_weights": [torch.rand(2, 1, 1)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 0)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 0)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]
                    ).float(),  # [end, node-add]
                    "event_src_logits": torch.rand(2, 1),
                    "event_dst_logits": torch.rand(2, 1),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[3, 0], [14, 3]]
                    ),  # [end, inventory]
                    "decoded_event_label_mask": torch.tensor(
                        [[True, False], [True, True]]
                    ),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 2, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True, True], [True, True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1]),
                        x=torch.tensor([[13, 3], [13, 3]]),
                        node_label_mask=torch.ones(2, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1]]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.ones(2, 1, 8),
                    "batch_node_mask": torch.tensor([[True], [True]]),
                    "self_attn_weights": [torch.rand(2, 1, 2)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 1)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 1)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0]]
                    ).float(),  # [edge-add, edge-add]
                    "event_src_logits": torch.tensor([[0, 1], [1, 0]]).float(),
                    "event_dst_logits": torch.tensor([[0, 1], [0, 1]]).float(),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[12, 3], [7, 3]]
                    ),  # [in, is]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 3, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True, True, True], [True, True, True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[13, 3], [13, 3], [14, 3]]),
                        node_label_mask=torch.ones(3, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.zeros(2, 2, 8),
                    "batch_node_mask": torch.tensor([[True, False], [True, True]]),
                    "self_attn_weights": [torch.rand(2, 1, 3)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 2)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 2)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]]
                    ).float(),  # [node-add, end]
                    "event_src_logits": torch.tensor([[0, 1], [0, 1]]).float(),
                    "event_dst_logits": torch.tensor([[0, 1], [1, 0]]).float(),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[13, 3], [14, 3]]
                    ),  # [player, inventory]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 4, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True, True, True, False], [True, True, True, True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[13, 3], [13, 3], [14, 3]]),
                        node_label_mask=torch.ones(3, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                        edge_index=torch.tensor([[1], [2]]),
                        edge_attr=torch.tensor([[7, 3]]),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3]]),
                    ),
                    "batch_node_embeddings": torch.ones(2, 2, 8),
                    "batch_node_mask": torch.tensor([[True, False], [True, True]]),
                    "self_attn_weights": [torch.rand(2, 1, 4)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 2)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 2)],
                },
            ],
            Batch(
                batch=torch.empty(0, dtype=torch.long),
                x=torch.empty(0, 0, dtype=torch.long),
                node_label_mask=torch.empty(0, 0).bool(),
                node_last_update=torch.empty(0, 2).long(),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=torch.empty(0, 0).long(),
                edge_label_mask=torch.empty(0, 0).bool(),
                edge_last_update=torch.empty(0, 2).long(),
            ),
            [
                {
                    "decoded_event_type_ids": torch.tensor(
                        [3, 3]
                    ),  # [node-add, node-add]
                    "decoded_event_src_ids": torch.zeros(2).long(),
                    "decoded_event_dst_ids": torch.zeros(2).long(),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[13, 3], [13, 3]]
                    ),  # [player, player]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_batched_graph": Batch(
                        batch=torch.empty(0).long(),
                        x=torch.empty(0, 0, dtype=torch.long),
                        node_label_mask=torch.empty(0, 0).bool(),
                        node_last_update=torch.empty(0, 2).long(),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.empty(2, 0, 8),
                    "batch_node_mask": torch.empty(2, 0).bool(),
                },
                {
                    "decoded_event_type_ids": torch.tensor([2, 3]),  # [end, node-add]
                    "decoded_event_src_ids": torch.zeros(2).long(),
                    "decoded_event_dst_ids": torch.zeros(2).long(),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[3, 0], [14, 3]]
                    ),  # [end, inventory]
                    "decoded_event_label_mask": torch.tensor(
                        [[True, False], [True, True]]
                    ),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1]),
                        x=torch.tensor([[13, 3], [13, 3]]),
                        node_label_mask=torch.ones(2, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1]]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.ones(2, 1, 8),
                    "batch_node_mask": torch.tensor([[True], [True]]),
                },
                {
                    "decoded_event_type_ids": torch.tensor([0, 5]),  # [pad, edge-add]
                    "decoded_event_src_ids": torch.tensor([0, 0]),
                    "decoded_event_dst_ids": torch.tensor([0, 1]),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[12, 3], [7, 3]]
                    ),  # [in, is]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[13, 3], [13, 3], [14, 3]]),
                        node_label_mask=torch.ones(3, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.zeros(2, 2, 8),
                    "batch_node_mask": torch.tensor([[True, False], [True, True]]),
                },
                {
                    "decoded_event_type_ids": torch.tensor([0, 2]),  # [pad, end]
                    "decoded_event_src_ids": torch.tensor([0, 1]),
                    "decoded_event_dst_ids": torch.tensor([0, 0]),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[13, 3], [14, 3]]
                    ),  # [player, inventory]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[13, 3], [13, 3], [14, 3]]),
                        node_label_mask=torch.ones(3, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                        edge_index=torch.tensor([[1], [2]]),
                        edge_attr=torch.tensor([[7, 3]]),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3]]),
                    ),
                    "batch_node_embeddings": torch.ones(2, 2, 8),
                    "batch_node_mask": torch.tensor([[True, False], [True, True]]),
                },
            ],
        ),
        (
            2,
            3,
            False,
            2,
            3,
            5,
            [
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]
                    ).float(),  # [node-add, node-add]
                    "event_src_logits": torch.empty(2, 0),
                    "event_dst_logits": torch.empty(2, 0),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[13, 3], [13, 3]]
                    ),  # [player, player]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 1, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True], [True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.empty(0).long(),
                        x=torch.empty(0, 0, dtype=torch.long),
                        node_label_mask=torch.empty(0, 0).bool(),
                        node_last_update=torch.empty(0, 2).long(),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.empty(2, 0, 8),
                    "batch_node_mask": torch.empty(2, 0).bool(),
                    "self_attn_weights": [torch.rand(2, 1, 1)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 0)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 0)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]
                    ).float(),  # [end, node-add]
                    "event_src_logits": torch.rand(2, 1),
                    "event_dst_logits": torch.rand(2, 1),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[3, 0], [14, 3]]
                    ),  # [end, inventory]
                    "decoded_event_label_mask": torch.tensor(
                        [[True, False], [True, True]]
                    ),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 2, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True, True], [True, True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1]),
                        x=torch.tensor([[13, 3], [13, 3]]),
                        node_label_mask=torch.ones(2, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1]]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.ones(2, 1, 8),
                    "batch_node_mask": torch.tensor([[True], [True]]),
                    "self_attn_weights": [torch.rand(2, 1, 2)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 1)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 1)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0]]
                    ).float(),  # [edge-add, edge-add]
                    "event_src_logits": torch.tensor([[0, 1], [1, 0]]).float(),
                    "event_dst_logits": torch.tensor([[0, 1], [0, 1]]).float(),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[12, 3], [7, 3]]
                    ),  # [in, is]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 3, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True, True, True], [True, True, True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[13, 3], [13, 3], [14, 3]]),
                        node_label_mask=torch.ones(3, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.zeros(2, 2, 8),
                    "batch_node_mask": torch.tensor([[True, False], [True, True]]),
                    "self_attn_weights": [torch.rand(2, 1, 3)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 2)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 2)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]]
                    ).float(),  # [node-add, end]
                    "event_src_logits": torch.tensor([[0, 1], [0, 1]]).float(),
                    "event_dst_logits": torch.tensor([[0, 1], [1, 0]]).float(),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[13, 3], [14, 3]]
                    ),  # [player, inventory]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 4, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True, True, True, False], [True, True, True, False]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[13, 3], [13, 3], [14, 3]]),
                        node_label_mask=torch.ones(3, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                        edge_index=torch.tensor([[1], [2]]),
                        edge_attr=torch.tensor([[7, 3]]),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3]]),
                    ),
                    "batch_node_embeddings": torch.ones(2, 2, 8),
                    "batch_node_mask": torch.tensor([[True, False], [True, True]]),
                    "self_attn_weights": [torch.rand(2, 1, 4)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 2)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 2)],
                },
            ],
            Batch(
                batch=torch.empty(0, dtype=torch.long),
                x=torch.empty(0, 0, dtype=torch.long),
                node_label_mask=torch.empty(0, 0).bool(),
                node_last_update=torch.empty(0, 2).long(),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=torch.empty(0, 0).long(),
                edge_label_mask=torch.empty(0, 0).bool(),
                edge_last_update=torch.empty(0, 2).long(),
            ),
            [
                {
                    "decoded_event_type_ids": torch.tensor(
                        [3, 3]
                    ),  # [node-add, node-add]
                    "decoded_event_src_ids": torch.zeros(2).long(),
                    "decoded_event_dst_ids": torch.zeros(2).long(),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[13, 3], [13, 3]]
                    ),  # [player, player]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_batched_graph": Batch(
                        batch=torch.empty(0).long(),
                        x=torch.empty(0, 0, dtype=torch.long),
                        node_label_mask=torch.empty(0, 0).bool(),
                        node_last_update=torch.empty(0, 2).long(),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.empty(2, 0, 8),
                    "batch_node_mask": torch.empty(2, 0).bool(),
                },
                {
                    "decoded_event_type_ids": torch.tensor([2, 3]),  # [end, node-add]
                    "decoded_event_src_ids": torch.zeros(2).long(),
                    "decoded_event_dst_ids": torch.zeros(2).long(),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[3, 0], [14, 3]]
                    ),  # [end, inventory]
                    "decoded_event_label_mask": torch.tensor(
                        [[True, False], [True, True]]
                    ),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1]),
                        x=torch.tensor([[13, 3], [13, 3]]),
                        node_label_mask=torch.ones(2, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1]]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.ones(2, 1, 8),
                    "batch_node_mask": torch.tensor([[True], [True]]),
                },
            ],
        ),
        (
            10,
            5,
            False,
            2,
            3,
            5,
            [
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]
                    ).float(),  # [node-add, node-add]
                    "event_src_logits": torch.empty(2, 0),
                    "event_dst_logits": torch.empty(2, 0),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[13, 3], [13, 3]]
                    ),  # [player, player]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 1, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True], [True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.empty(0).long(),
                        x=torch.empty(0, 0, dtype=torch.long),
                        node_label_mask=torch.empty(0, 0).bool(),
                        node_last_update=torch.empty(0, 2).long(),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.empty(2, 0, 8),
                    "batch_node_mask": torch.empty(2, 0).bool(),
                    "self_attn_weights": [torch.rand(2, 1, 1)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 0)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 0)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]
                    ).float(),  # [end, node-add]
                    "event_src_logits": torch.rand(2, 1),
                    "event_dst_logits": torch.rand(2, 1),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[3, 0], [14, 3]]
                    ),  # [end, inventory]
                    "decoded_event_label_mask": torch.tensor(
                        [[True, False], [True, True]]
                    ),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 2, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True, True], [True, True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1]),
                        x=torch.tensor([[13, 3], [13, 3]]),
                        node_label_mask=torch.ones(2, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1]]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.ones(2, 1, 8),
                    "batch_node_mask": torch.tensor([[True], [True]]),
                    "self_attn_weights": [torch.rand(2, 1, 2)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 1)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 1)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1, 0]]
                    ).float(),  # [edge-delete, edge-add]
                    "event_src_logits": torch.tensor([[0, 1], [0, 1]]).float(),
                    "event_dst_logits": torch.tensor([[1, 0], [1, 0]]).float(),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[7, 3], [12, 3]]
                    ),  # [is, in]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 3, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True, True, True], [True, True, True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[13, 3], [13, 3], [14, 3]]),
                        node_label_mask=torch.ones(3, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.zeros(2, 2, 8),
                    "batch_node_mask": torch.tensor([[True, False], [True, True]]),
                    "self_attn_weights": [torch.rand(2, 1, 3)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 2)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 2)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1]]
                    ).float(),  # [edge-delete, edge-delete]
                    "event_src_logits": torch.tensor([[1, 0], [0, 1]]).float(),
                    "event_dst_logits": torch.tensor([[0, 0], [1, 0]]).float(),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[7, 3], [12, 3]]
                    ),  # [is, in]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 4, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True, True, True, False], [True, True, True, True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[13, 3], [13, 3], [14, 3]]),
                        node_label_mask=torch.ones(3, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                        edge_index=torch.tensor([[2], [1]]),
                        edge_attr=torch.tensor([[12, 3]]),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3]]),
                    ),
                    "batch_node_embeddings": torch.ones(2, 2, 8),
                    "batch_node_mask": torch.tensor([[True, False], [True, True]]),
                    "self_attn_weights": [torch.rand(2, 1, 4)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 2)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 2)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0, 0]]
                    ).float(),  # [edge-delete, node-delete]
                    "event_src_logits": torch.tensor([[0, 1], [0, 1]]).float(),
                    "event_dst_logits": torch.tensor([[1, 0], [0, 1]]).float(),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[13, 3], [13, 3]]
                    ),  # [player, player]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 5, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [
                            [True, True, True, False, False],
                            [True, True, True, True, True],
                        ]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[13, 3], [13, 3], [14, 3]]),
                        node_label_mask=torch.ones(3, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.zeros(2, 2, 8),
                    "batch_node_mask": torch.tensor([[True, False], [True, True]]),
                    "self_attn_weights": [torch.rand(2, 1, 5)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 2)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 2)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]]
                    ).float(),  # [node-add, end]
                    "event_src_logits": torch.tensor([[0, 1], [1, 0]]).float(),
                    "event_dst_logits": torch.tensor([[1, 0], [0, 1]]).float(),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[13, 3], [14, 3]]
                    ),  # [player, inventory]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 6, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [
                            [True, True, True, False, False, False],
                            [True, True, True, True, True, True],
                        ]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[13, 3], [13, 3], [14, 3]]),
                        node_label_mask=torch.ones(3, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.ones(2, 2, 8),
                    "batch_node_mask": torch.tensor([[True, False], [True, True]]),
                    "self_attn_weights": [torch.rand(2, 1, 6)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 2)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 2)],
                },
            ],
            Batch(
                batch=torch.empty(0, dtype=torch.long),
                x=torch.empty(0, 0, dtype=torch.long),
                node_label_mask=torch.empty(0, 0).bool(),
                node_last_update=torch.empty(0, 2, dtype=torch.long),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=torch.empty(0, 0).long(),
                edge_label_mask=torch.empty(0, 0).bool(),
                edge_last_update=torch.empty(0, 2, dtype=torch.long),
            ),
            [
                {
                    "decoded_event_type_ids": torch.tensor(
                        [3, 3]
                    ),  # [node-add, node-add]
                    "decoded_event_src_ids": torch.zeros(2).long(),
                    "decoded_event_dst_ids": torch.zeros(2).long(),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[13, 3], [13, 3]]
                    ),  # [player, player]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_batched_graph": Batch(
                        batch=torch.empty(0).long(),
                        x=torch.empty(0, 0, dtype=torch.long),
                        node_label_mask=torch.empty(0, 0).bool(),
                        node_last_update=torch.empty(0, 2).long(),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.empty(2, 0, 8),
                    "batch_node_mask": torch.empty(2, 0).bool(),
                },
                {
                    "decoded_event_type_ids": torch.tensor([2, 3]),  # [end, node-add]
                    "decoded_event_src_ids": torch.tensor([0, 0]),
                    "decoded_event_dst_ids": torch.tensor([0, 0]),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[3, 0], [14, 3]]
                    ),  # [end, inventory]
                    "decoded_event_label_mask": torch.tensor(
                        [[True, False], [True, True]]
                    ),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1]),
                        x=torch.tensor([[13, 3], [13, 3]]),
                        node_label_mask=torch.ones(2, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1]]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.ones(2, 1, 8),
                    "batch_node_mask": torch.tensor([[True], [True]]),
                },
                {
                    "decoded_event_type_ids": torch.tensor([0, 5]),  # [pad, edge-add]
                    "decoded_event_src_ids": torch.tensor([0, 1]),
                    "decoded_event_dst_ids": torch.tensor([0, 0]),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[7, 3], [12, 3]]
                    ),  # [is, in]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[13, 3], [13, 3], [14, 3]]),
                        node_label_mask=torch.ones(3, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.zeros(2, 2, 8),
                    "batch_node_mask": torch.tensor([[True, False], [True, True]]),
                },
                {
                    "decoded_event_type_ids": torch.tensor(
                        [0, 6]
                    ),  # [pad, edge-delete]
                    "decoded_event_src_ids": torch.tensor([0, 1]),
                    "decoded_event_dst_ids": torch.tensor([0, 0]),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[7, 3], [12, 3]]
                    ),  # [is, in]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[13, 3], [13, 3], [14, 3]]),
                        node_label_mask=torch.ones(3, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                        edge_index=torch.tensor([[2], [1]]),
                        edge_attr=torch.tensor([[12, 3]]),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3]]),
                    ),
                    "batch_node_embeddings": torch.ones(2, 2, 8),
                    "batch_node_mask": torch.tensor([[True, False], [True, True]]),
                },
                {
                    "decoded_event_type_ids": torch.tensor(
                        [0, 4]
                    ),  # [pad, node-delete]
                    "decoded_event_src_ids": torch.tensor([0, 1]),
                    "decoded_event_dst_ids": torch.tensor([0, 1]),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[13, 3], [13, 3]]
                    ),  # [player, player]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[13, 3], [13, 3], [14, 3]]),
                        node_label_mask=torch.ones(3, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.zeros(2, 2, 8),
                    "batch_node_mask": torch.tensor([[True, False], [True, True]]),
                },
                {
                    "decoded_event_type_ids": torch.tensor([0, 2]),  # [pad, end]
                    "decoded_event_src_ids": torch.tensor([0, 0]),
                    "decoded_event_dst_ids": torch.tensor([0, 1]),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[13, 3], [14, 3]]
                    ),  # [player, inventory]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[13, 3], [13, 3], [14, 3]]),
                        node_label_mask=torch.ones(3, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                        edge_index=torch.empty(2, 0).long(),
                        edge_attr=torch.empty(0, 0).long(),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2).long(),
                    ),
                    "batch_node_embeddings": torch.ones(2, 2, 8),
                    "batch_node_mask": torch.tensor([[True, False], [True, True]]),
                },
            ],
        ),
        (
            10,
            4,
            False,
            2,
            3,
            5,
            [
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]
                    ).float(),  # [node-add, node-add]
                    "event_src_logits": torch.tensor([[0, 1], [1, 0]]).float(),
                    "event_dst_logits": torch.tensor([[0, 1], [0, 1]]).float(),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[13, 3], [13, 3]]
                    ),  # [player, player]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 1, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True], [True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[13, 3], [13, 3], [14, 3]]),
                        node_label_mask=torch.ones(3, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                        edge_index=torch.tensor([[1], [2]]),
                        edge_attr=torch.tensor([[12, 3]]),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3]]),
                    ),
                    "batch_node_embeddings": torch.ones(2, 2, 8),
                    "batch_node_mask": torch.tensor([[True, False], [True, True]]),
                    "self_attn_weights": [torch.rand(2, 1, 1)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 2)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 2)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]
                    ).float(),  # [end, node-add]
                    "event_src_logits": torch.tensor([[0, 1, 0], [0, 0, 1]]).float(),
                    "event_dst_logits": torch.tensor([[0, 1, 0], [0, 0, 1]]).float(),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[3, 0], [14, 3]]
                    ),  # [end, inventory]
                    "decoded_event_label_mask": torch.tensor(
                        [[True, False], [True, True]]
                    ),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 2, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True, True], [True, True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1]),
                        x=torch.tensor([[13, 3], [13, 3], [13, 3], [14, 3], [13, 3]]),
                        node_label_mask=torch.ones(5, 2).bool(),
                        node_last_update=torch.tensor(
                            [[1, 2], [2, 0], [2, 1], [2, 2], [3, 0]]
                        ),
                        edge_index=torch.tensor([[2], [3]]),
                        edge_attr=torch.tensor([[12, 3]]),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3]]),
                    ),
                    "batch_node_embeddings": torch.ones(2, 3, 8),
                    "batch_node_mask": torch.tensor(
                        [[True, True, False], [True, True, True]]
                    ),
                    "self_attn_weights": [torch.rand(2, 1, 2)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 3)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 3)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0]]
                    ).float(),  # [edge-add, edge-add]
                    "event_src_logits": torch.tensor(
                        [[0, 1, 0, 0], [0, 0, 1, 0]]
                    ).float(),
                    "event_dst_logits": torch.tensor(
                        [[0, 1, 0, 0], [0, 0, 0, 1]]
                    ).float(),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[12, 3], [7, 3]]
                    ),  # [in, is]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 3, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True, True, True], [True, True, True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1, 1]),
                        x=torch.tensor(
                            [[13, 3], [13, 3], [13, 3], [14, 3], [13, 3], [14, 3]]
                        ),
                        node_label_mask=torch.ones(6, 2).bool(),
                        node_last_update=torch.tensor(
                            [[1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1]]
                        ),
                        edge_index=torch.tensor([[2], [3]]),
                        edge_attr=torch.tensor([[12, 3]]),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3]]),
                    ),
                    "batch_node_embeddings": torch.ones(2, 4, 8),
                    "batch_node_mask": torch.tensor(
                        [[True, True, False, False], [True, True, True, True]]
                    ),
                    "self_attn_weights": [torch.rand(2, 1, 3)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 4)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 4)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]]
                    ).float(),  # [node-add, end]
                    "event_src_logits": torch.tensor(
                        [[0, 1, 0, 0], [0, 0, 1, 0]]
                    ).float(),
                    "event_dst_logits": torch.tensor(
                        [[0, 1, 0, 0], [0, 0, 0, 1]]
                    ).float(),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[13, 3], [14, 3]]
                    ),  # [player, inventory]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 4, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True, True, True, False], [True, True, True, True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1, 1]),
                        x=torch.tensor(
                            [[13, 3], [13, 3], [13, 3], [14, 3], [13, 3], [14, 3]]
                        ),
                        node_label_mask=torch.ones(6, 2).bool(),
                        node_last_update=torch.tensor(
                            [[1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1]]
                        ),
                        edge_index=torch.tensor([[2, 5], [3, 6]]),
                        edge_attr=torch.tensor([[12, 3], [12, 3]]),
                        edge_label_mask=torch.ones(2, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3], [3, 2]]),
                    ),
                    "batch_node_embeddings": torch.zeros(2, 4, 8),
                    "batch_node_mask": torch.tensor(
                        [[True, True, False, False], [True, True, True, True]]
                    ),
                    "self_attn_weights": [torch.rand(2, 1, 4)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 4)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 4)],
                },
            ],
            Batch(
                batch=torch.tensor([0, 1, 1]),
                x=torch.tensor([[13, 3], [13, 3], [14, 3]]),
                node_label_mask=torch.ones(3, 2).bool(),
                node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                edge_index=torch.tensor([[1], [2]]),
                edge_attr=torch.tensor([[12, 3]]),
                edge_label_mask=torch.ones(1, 2).bool(),
                edge_last_update=torch.tensor([[2, 3]]),
            ),
            [
                {
                    "decoded_event_type_ids": torch.tensor(
                        [3, 3]
                    ),  # [node-add, node-add]
                    "decoded_event_src_ids": torch.tensor([0, 0]),
                    "decoded_event_dst_ids": torch.tensor([0, 1]),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[13, 3], [13, 3]]
                    ),  # [player, player]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[13, 3], [13, 3], [14, 3]]),
                        node_label_mask=torch.ones(3, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                        edge_index=torch.tensor([[1], [2]]),
                        edge_attr=torch.tensor([[12, 3]]),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3]]),
                    ),
                    "batch_node_embeddings": torch.ones(2, 2, 8),
                    "batch_node_mask": torch.tensor([[True, False], [True, True]]),
                },
                {
                    "decoded_event_type_ids": torch.tensor([2, 3]),  # [end, node-add]
                    "decoded_event_src_ids": torch.tensor([1, 2]),
                    "decoded_event_dst_ids": torch.tensor([1, 2]),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[3, 0], [14, 3]]
                    ),  # [end, inventory]
                    "decoded_event_label_mask": torch.tensor(
                        [[True, False], [True, True]]
                    ),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1]),
                        x=torch.tensor([[13, 3], [13, 3], [13, 3], [14, 3], [13, 3]]),
                        node_label_mask=torch.ones(5, 2).bool(),
                        node_last_update=torch.tensor(
                            [[1, 2], [2, 0], [2, 1], [2, 2], [3, 0]]
                        ),
                        edge_index=torch.tensor([[2], [3]]),
                        edge_attr=torch.tensor([[12, 3]]),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3]]),
                    ),
                    "batch_node_embeddings": torch.ones(2, 3, 8),
                    "batch_node_mask": torch.tensor(
                        [[True, True, False], [True, True, True]]
                    ),
                },
                {
                    "decoded_event_type_ids": torch.tensor([0, 5]),  # [pad, edge-add]
                    "decoded_event_src_ids": torch.tensor([1, 2]),
                    "decoded_event_dst_ids": torch.tensor([1, 3]),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[12, 3], [7, 3]]
                    ),  # [in, is]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1, 1]),
                        x=torch.tensor(
                            [[13, 3], [13, 3], [13, 3], [14, 3], [13, 3], [14, 3]]
                        ),
                        node_label_mask=torch.ones(6, 2).bool(),
                        node_last_update=torch.tensor(
                            [[1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1]]
                        ),
                        edge_index=torch.tensor([[2], [3]]),
                        edge_attr=torch.tensor([[12, 3]]),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3]]),
                    ),
                    "batch_node_embeddings": torch.ones(2, 4, 8),
                    "batch_node_mask": torch.tensor(
                        [[True, True, False, False], [True, True, True, True]]
                    ),
                },
                {
                    "decoded_event_type_ids": torch.tensor([0, 2]),  # [pad, end]
                    "decoded_event_src_ids": torch.tensor([1, 2]),
                    "decoded_event_dst_ids": torch.tensor([1, 3]),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[13, 3], [14, 3]]
                    ),  # [player, inventory]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1, 1]),
                        x=torch.tensor(
                            [[13, 3], [13, 3], [13, 3], [14, 3], [13, 3], [14, 3]]
                        ),
                        node_label_mask=torch.ones(6, 2).bool(),
                        node_last_update=torch.tensor(
                            [[1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1]]
                        ),
                        edge_index=torch.tensor([[2, 5], [3, 6]]),
                        edge_attr=torch.tensor([[12, 3], [12, 3]]),
                        edge_label_mask=torch.ones(2, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3], [3, 2]]),
                    ),
                    "batch_node_embeddings": torch.zeros(2, 4, 8),
                    "batch_node_mask": torch.tensor(
                        [[True, True, False, False], [True, True, True, True]]
                    ),
                },
            ],
        ),
        (
            2,
            5,
            False,
            2,
            3,
            5,
            [
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]
                    ).float(),  # [node-add, node-add]
                    "event_src_logits": torch.tensor([[0, 1], [1, 0]]).float(),
                    "event_dst_logits": torch.tensor([[0, 1], [0, 1]]).float(),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[13, 3], [13, 3]]
                    ),  # [player, player]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 1, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True], [True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[13, 3], [13, 3], [14, 3]]),
                        node_label_mask=torch.ones(3, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                        edge_index=torch.tensor([[1], [2]]),
                        edge_attr=torch.tensor([[12, 3]]),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3]]),
                    ),
                    "batch_node_embeddings": torch.ones(2, 2, 8),
                    "batch_node_mask": torch.tensor([[True, False], [True, True]]),
                    "self_attn_weights": [torch.rand(2, 1, 1)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 2)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 2)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]
                    ).float(),  # [end, node-add]
                    "event_src_logits": torch.tensor([[0, 1, 0], [0, 0, 1]]).float(),
                    "event_dst_logits": torch.tensor([[0, 1, 0], [0, 0, 1]]).float(),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[3, 0], [14, 3]]
                    ),  # [end, inventory]
                    "decoded_event_label_mask": torch.tensor(
                        [[True, False], [True, True]]
                    ),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 2, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True, True], [True, True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1]),
                        x=torch.tensor([[13, 3], [13, 3], [13, 3], [14, 3], [13, 3]]),
                        node_label_mask=torch.ones(5, 2).bool(),
                        node_last_update=torch.tensor(
                            [[1, 2], [2, 0], [2, 1], [2, 2], [3, 0]]
                        ),
                        edge_index=torch.tensor([[2], [3]]),
                        edge_attr=torch.tensor([[12, 3]]),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3]]),
                    ),
                    "batch_node_embeddings": torch.ones(2, 3, 8),
                    "batch_node_mask": torch.tensor(
                        [[True, True, False], [True, True, True]]
                    ),
                    "self_attn_weights": [torch.rand(2, 1, 2)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 3)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 3)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0]]
                    ).float(),  # [edge-add, edge-add]
                    "event_src_logits": torch.tensor(
                        [[0, 1, 0, 0], [0, 0, 1, 0]]
                    ).float(),
                    "event_dst_logits": torch.tensor(
                        [[0, 1, 0, 0], [0, 0, 0, 1]]
                    ).float(),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[12, 3], [7, 3]]
                    ),  # [in, is]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 3, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True, True, True], [True, True, True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1, 1]),
                        x=torch.tensor(
                            [[13, 3], [13, 3], [13, 3], [14, 3], [13, 3], [14, 3]]
                        ),
                        node_label_mask=torch.ones(6, 2).bool(),
                        node_last_update=torch.tensor(
                            [[1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1]]
                        ),
                        edge_index=torch.tensor([[2], [3]]),
                        edge_attr=torch.tensor([[12, 3]]),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3]]),
                    ),
                    "batch_node_embeddings": torch.ones(2, 4, 8),
                    "batch_node_mask": torch.tensor(
                        [[True, True, False, False], [True, True, True, True]]
                    ),
                    "self_attn_weights": [torch.rand(2, 1, 3)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 4)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 4)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]]
                    ).float(),  # [node-add, end]
                    "event_src_logits": torch.tensor(
                        [[0, 1, 0, 0], [0, 0, 1, 0]]
                    ).float(),
                    "event_dst_logits": torch.tensor(
                        [[0, 1, 0, 0], [0, 0, 0, 1]]
                    ).float(),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[13, 3], [14, 3]]
                    ),  # [player, inventory]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 4, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True, True, True, False], [True, True, True, True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1, 1]),
                        x=torch.tensor(
                            [[13, 3], [13, 3], [13, 3], [14, 3], [13, 3], [14, 3]]
                        ),
                        node_label_mask=torch.ones(6, 2).bool(),
                        node_last_update=torch.tensor(
                            [[1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1]]
                        ),
                        edge_index=torch.tensor([[2, 5], [3, 6]]),
                        edge_attr=torch.tensor([[12, 3], [12, 3]]),
                        edge_label_mask=torch.ones(2, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3], [3, 2]]),
                    ),
                    "batch_node_embeddings": torch.zeros(2, 4, 8),
                    "batch_node_mask": torch.tensor(
                        [[True, True, False, False], [True, True, True, True]]
                    ),
                    "self_attn_weights": [torch.rand(2, 1, 4)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 4)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 4)],
                },
            ],
            Batch(
                batch=torch.tensor([0, 1, 1]),
                x=torch.tensor([[13, 3], [13, 3], [14, 3]]),
                node_label_mask=torch.ones(3, 2).bool(),
                node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                edge_index=torch.tensor([[1], [2]]),
                edge_attr=torch.tensor([[12, 3]]),
                edge_label_mask=torch.ones(1, 2).bool(),
                edge_last_update=torch.tensor([[2, 3]]),
            ),
            [
                {
                    "decoded_event_type_ids": torch.tensor(
                        [3, 3]
                    ),  # [node-add, node-add]
                    "decoded_event_src_ids": torch.tensor([0, 0]),
                    "decoded_event_dst_ids": torch.tensor([0, 1]),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[13, 3], [13, 3]]
                    ),  # [player, player]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=torch.tensor([[13, 3], [13, 3], [14, 3]]),
                        node_label_mask=torch.ones(3, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                        edge_index=torch.tensor([[1], [2]]),
                        edge_attr=torch.tensor([[12, 3]]),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3]]),
                    ),
                    "batch_node_embeddings": torch.ones(2, 2, 8),
                    "batch_node_mask": torch.tensor([[True, False], [True, True]]),
                },
                {
                    "decoded_event_type_ids": torch.tensor([2, 3]),  # [end, node-add]
                    "decoded_event_src_ids": torch.tensor([1, 2]),
                    "decoded_event_dst_ids": torch.tensor([1, 2]),
                    "decoded_event_label_word_ids": torch.tensor(
                        [[3, 0], [14, 3]]
                    ),  # [end, inventory]
                    "decoded_event_label_mask": torch.tensor(
                        [[True, False], [True, True]]
                    ),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1]),
                        x=torch.tensor([[13, 3], [13, 3], [13, 3], [14, 3], [13, 3]]),
                        node_label_mask=torch.ones(5, 2).bool(),
                        node_last_update=torch.tensor(
                            [[1, 2], [2, 0], [2, 1], [2, 2], [3, 0]]
                        ),
                        edge_index=torch.tensor([[2], [3]]),
                        edge_attr=torch.tensor([[12, 3]]),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3]]),
                    ),
                    "batch_node_embeddings": torch.ones(2, 3, 8),
                    "batch_node_mask": torch.tensor(
                        [[True, True, False], [True, True, True]]
                    ),
                },
            ],
        ),
        (
            10,
            4,
            True,
            2,
            3,
            5,
            [
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]
                    ).float(),  # [node-add, node-add]
                    "event_src_logits": torch.tensor([[0, 1], [1, 0]]).float(),
                    "event_dst_logits": torch.tensor([[0, 1], [0, 1]]).float(),
                    "decoded_event_label_word_ids": F.one_hot(
                        torch.tensor([[13, 3], [13, 3]]), num_classes=20
                    ),  # [player, player]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 1, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True], [True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=F.one_hot(
                            torch.tensor([[13, 3], [13, 3], [14, 3]]), num_classes=20
                        ).float(),
                        node_label_mask=torch.ones(3, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                        edge_index=torch.tensor([[1], [2]]),
                        edge_attr=F.one_hot(
                            torch.tensor([[12, 3]]), num_classes=20
                        ).float(),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3]]),
                    ),
                    "batch_node_embeddings": torch.ones(2, 2, 8),
                    "batch_node_mask": torch.tensor([[True, False], [True, True]]),
                    "self_attn_weights": [torch.rand(2, 1, 1)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 2)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 2)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]
                    ).float(),  # [end, node-add]
                    "event_src_logits": torch.tensor([[0, 1, 0], [0, 0, 1]]).float(),
                    "event_dst_logits": torch.tensor([[0, 1, 0], [0, 0, 1]]).float(),
                    "decoded_event_label_word_ids": F.one_hot(
                        torch.tensor([[3, 0], [14, 3]]), num_classes=20
                    ),  # [end, inventory]
                    "decoded_event_label_mask": torch.tensor(
                        [[True, False], [True, True]]
                    ),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 2, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True, True], [True, True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1]),
                        x=F.one_hot(
                            torch.tensor([[13, 3], [13, 3], [13, 3], [14, 3], [13, 3]]),
                            num_classes=20,
                        ).float(),
                        node_label_mask=torch.ones(5, 2).bool(),
                        node_last_update=torch.tensor(
                            [[1, 2], [2, 0], [2, 1], [2, 2], [3, 0]]
                        ),
                        edge_index=torch.tensor([[2], [3]]),
                        edge_attr=F.one_hot(
                            torch.tensor([[12, 3]]), num_classes=20
                        ).float(),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3]]),
                    ),
                    "batch_node_embeddings": torch.ones(2, 3, 8),
                    "batch_node_mask": torch.tensor(
                        [[True, True, False], [True, True, True]]
                    ),
                    "self_attn_weights": [torch.rand(2, 1, 2)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 3)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 3)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0]]
                    ).float(),  # [edge-add, edge-add]
                    "event_src_logits": torch.tensor(
                        [[0, 1, 0, 0], [0, 0, 1, 0]]
                    ).float(),
                    "event_dst_logits": torch.tensor(
                        [[0, 1, 0, 0], [0, 0, 0, 1]]
                    ).float(),
                    "decoded_event_label_word_ids": F.one_hot(
                        torch.tensor([[12, 3], [7, 3]]), num_classes=20
                    ),  # [in, is]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 3, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True, True, True], [True, True, True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1, 1]),
                        x=F.one_hot(
                            torch.tensor(
                                [[13, 3], [13, 3], [13, 3], [14, 3], [13, 3], [14, 3]]
                            ),
                            num_classes=20,
                        ).float(),
                        node_label_mask=torch.ones(6, 2).bool(),
                        node_last_update=torch.tensor(
                            [[1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1]]
                        ),
                        edge_index=torch.tensor([[2], [3]]),
                        edge_attr=F.one_hot(
                            torch.tensor([[12, 3]]), num_classes=20
                        ).float(),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3]]),
                    ),
                    "batch_node_embeddings": torch.ones(2, 4, 8),
                    "batch_node_mask": torch.tensor(
                        [[True, True, False, False], [True, True, True, True]]
                    ),
                    "self_attn_weights": [torch.rand(2, 1, 3)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 4)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 4)],
                },
                {
                    "event_type_logits": torch.tensor(
                        [[0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]]
                    ).float(),  # [node-add, end]
                    "event_src_logits": torch.tensor(
                        [[0, 1, 0, 0], [0, 0, 1, 0]]
                    ).float(),
                    "event_dst_logits": torch.tensor(
                        [[0, 1, 0, 0], [0, 0, 0, 1]]
                    ).float(),
                    "decoded_event_label_word_ids": F.one_hot(
                        torch.tensor([[13, 3], [14, 3]]), num_classes=20
                    ),  # [player, inventory]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_prev_input_event_emb_seq": torch.rand(1, 2, 4, 8),
                    "updated_prev_input_event_emb_seq_mask": torch.tensor(
                        [[True, True, True, False], [True, True, True, True]]
                    ),
                    "encoded_obs": torch.rand(2, 3, 8),
                    "encoded_prev_action": torch.rand(2, 5, 8),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1, 1]),
                        x=F.one_hot(
                            torch.tensor(
                                [[13, 3], [13, 3], [13, 3], [14, 3], [13, 3], [14, 3]]
                            ),
                            num_classes=20,
                        ).float(),
                        node_label_mask=torch.ones(6, 2).bool(),
                        node_last_update=torch.tensor(
                            [[1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1]]
                        ),
                        edge_index=torch.tensor([[2, 5], [3, 6]]),
                        edge_attr=F.one_hot(
                            torch.tensor([[12, 3], [12, 3]]), num_classes=20
                        ).float(),
                        edge_label_mask=torch.ones(2, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3], [3, 2]]),
                    ),
                    "batch_node_embeddings": torch.zeros(2, 4, 8),
                    "batch_node_mask": torch.tensor(
                        [[True, True, False, False], [True, True, True, True]]
                    ),
                    "self_attn_weights": [torch.rand(2, 1, 4)],
                    "obs_graph_attn_weights": [torch.rand(2, 1, 3)],
                    "prev_action_graph_attn_weights": [torch.rand(2, 1, 5)],
                    "graph_obs_attn_weights": [torch.rand(2, 1, 4)],
                    "graph_prev_action_attn_weights": [torch.rand(2, 1, 4)],
                },
            ],
            Batch(
                batch=torch.tensor([0, 1, 1]),
                x=F.one_hot(
                    torch.tensor([[13, 3], [13, 3], [14, 3]]), num_classes=20
                ).float(),
                node_label_mask=torch.ones(3, 2).bool(),
                node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                edge_index=torch.tensor([[1], [2]]),
                edge_attr=F.one_hot(torch.tensor([[12, 3]]), num_classes=20).float(),
                edge_label_mask=torch.ones(1, 2).bool(),
                edge_last_update=torch.tensor([[2, 3]]),
            ),
            [
                {
                    "decoded_event_type_ids": F.one_hot(
                        torch.tensor([3, 3]), num_classes=len(EVENT_TYPES)
                    ).float(),  # [node-add, node-add]
                    "decoded_event_src_ids": F.one_hot(
                        torch.tensor([0, 0]), num_classes=2
                    ).float(),
                    "decoded_event_dst_ids": F.one_hot(
                        torch.tensor([0, 1]), num_classes=2
                    ).float(),
                    "decoded_event_label_word_ids": F.one_hot(
                        torch.tensor([[13, 3], [13, 3]]), num_classes=20
                    ),  # [player, player]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 1, 1]),
                        x=F.one_hot(
                            torch.tensor([[13, 3], [13, 3], [14, 3]]), num_classes=20
                        ).float(),
                        node_label_mask=torch.ones(3, 2).bool(),
                        node_last_update=torch.tensor([[1, 2], [2, 1], [2, 2]]),
                        edge_index=torch.tensor([[1], [2]]),
                        edge_attr=F.one_hot(
                            torch.tensor([[12, 3]]), num_classes=20
                        ).float(),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3]]),
                    ),
                    "batch_node_embeddings": torch.ones(2, 2, 8),
                    "batch_node_mask": torch.tensor([[True, False], [True, True]]),
                },
                {
                    "decoded_event_type_ids": F.one_hot(
                        torch.tensor([2, 3]), num_classes=len(EVENT_TYPES)
                    ).float(),  # [end, node-add]
                    "decoded_event_src_ids": F.one_hot(
                        torch.tensor([1, 2]), num_classes=3
                    ).float(),
                    "decoded_event_dst_ids": F.one_hot(
                        torch.tensor([1, 2]), num_classes=3
                    ).float(),
                    "decoded_event_label_word_ids": F.one_hot(
                        torch.tensor([[3, 0], [14, 3]]), num_classes=20
                    ),  # [end, inventory]
                    "decoded_event_label_mask": torch.tensor(
                        [[True, False], [True, True]]
                    ),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1]),
                        x=F.one_hot(
                            torch.tensor([[13, 3], [13, 3], [13, 3], [14, 3], [13, 3]]),
                            num_classes=20,
                        ).float(),
                        node_label_mask=torch.ones(5, 2).bool(),
                        node_last_update=torch.tensor(
                            [[1, 2], [2, 0], [2, 1], [2, 2], [3, 0]]
                        ),
                        edge_index=torch.tensor([[2], [3]]),
                        edge_attr=F.one_hot(
                            torch.tensor([[12, 3]]), num_classes=20
                        ).float(),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3]]),
                    ),
                    "batch_node_embeddings": torch.ones(2, 3, 8),
                    "batch_node_mask": torch.tensor(
                        [[True, True, False], [True, True, True]]
                    ),
                },
                {
                    "decoded_event_type_ids": torch.cat(
                        [
                            torch.zeros(1, len(EVENT_TYPES)),
                            F.one_hot(
                                torch.tensor([5]), num_classes=len(EVENT_TYPES)
                            ).float(),
                        ]
                    ),  # [pad, edge-add]
                    "decoded_event_src_ids": F.one_hot(
                        torch.tensor([1, 2]), num_classes=4
                    ).float(),
                    "decoded_event_dst_ids": F.one_hot(
                        torch.tensor([1, 3]), num_classes=4
                    ).float(),
                    "decoded_event_label_word_ids": F.one_hot(
                        torch.tensor([[12, 3], [7, 3]]), num_classes=20
                    ),  # [in, is]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1, 1]),
                        x=F.one_hot(
                            torch.tensor(
                                [[13, 3], [13, 3], [13, 3], [14, 3], [13, 3], [14, 3]]
                            ),
                            num_classes=20,
                        ).float(),
                        node_label_mask=torch.ones(6, 2).bool(),
                        node_last_update=torch.tensor(
                            [[1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1]]
                        ),
                        edge_index=torch.tensor([[2], [3]]),
                        edge_attr=F.one_hot(
                            torch.tensor([[12, 3]]), num_classes=20
                        ).float(),
                        edge_label_mask=torch.ones(1, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3]]),
                    ),
                    "batch_node_embeddings": torch.ones(2, 4, 8),
                    "batch_node_mask": torch.tensor(
                        [[True, True, False, False], [True, True, True, True]]
                    ),
                },
                {
                    "decoded_event_type_ids": torch.cat(
                        [
                            torch.zeros(1, len(EVENT_TYPES)),
                            F.one_hot(
                                torch.tensor([2]), num_classes=len(EVENT_TYPES)
                            ).float(),
                        ]
                    ),  # [pad, end]
                    "decoded_event_src_ids": F.one_hot(
                        torch.tensor([1, 2]), num_classes=4
                    ).float(),
                    "decoded_event_dst_ids": F.one_hot(
                        torch.tensor([1, 3]), num_classes=4
                    ).float(),
                    "decoded_event_label_word_ids": F.one_hot(
                        torch.tensor([[13, 3], [14, 3]]), num_classes=20
                    ),  # [player, inventory]
                    "decoded_event_label_mask": torch.ones(2, 2).bool(),
                    "updated_batched_graph": Batch(
                        batch=torch.tensor([0, 0, 1, 1, 1, 1]),
                        x=F.one_hot(
                            torch.tensor(
                                [[13, 3], [13, 3], [13, 3], [14, 3], [13, 3], [14, 3]]
                            ),
                            num_classes=20,
                        ).float(),
                        node_label_mask=torch.ones(6, 2).bool(),
                        node_last_update=torch.tensor(
                            [[1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1]]
                        ),
                        edge_index=torch.tensor([[2, 5], [3, 6]]),
                        edge_attr=F.one_hot(
                            torch.tensor([[12, 3], [12, 3]]), num_classes=20
                        ).float(),
                        edge_label_mask=torch.ones(2, 2).bool(),
                        edge_last_update=torch.tensor([[2, 3], [3, 2]]),
                    ),
                    "batch_node_embeddings": torch.zeros(2, 4, 8),
                    "batch_node_mask": torch.tensor(
                        [[True, True, False, False], [True, True, True, True]]
                    ),
                },
            ],
        ),
    ],
)
def test_tdgu_greedy_decode(
    monkeypatch,
    max_event_decode_len,
    max_label_decode_len,
    gumbel_greedy_decode,
    batch,
    obs_len,
    prev_action_len,
    forward_results,
    prev_batched_graph,
    expected_decoded_list,
    gumbel_tau,
):
    # monkeypatch gumbel_softmax to argmax() + F.one_hot()
    # to remove randomness for tests
    def mock_gumbel_softmax(logits, **kwargs):
        return F.one_hot(logits.argmax(-1), num_classes=logits.size(-1)).float()

    monkeypatch.setattr("tdgu.train.common.F.gumbel_softmax", mock_gumbel_softmax)

    # monkeypatch masked_gumbel_softmax to masked_softmax() + argmax() + F.one_hot()
    # to remove randomness for tests
    def mock_masked_gumbel_softmax(logits, mask, **kwargs):
        return F.one_hot(
            masked_softmax(logits, mask, dim=-1).argmax(-1), num_classes=logits.size(-1)
        ).float()

    monkeypatch.setattr(
        "tdgu.train.common.masked_gumbel_softmax", mock_masked_gumbel_softmax
    )

    tdgu = TDGULightningModule()

    class MockForward:
        def __init__(self):
            self.num_calls = 0

        def __call__(self, *args, **kwargs):
            decoded = forward_results[self.num_calls]
            decoded["encoded_obs"] = torch.rand(batch, obs_len, tdgu.hparams.hidden_dim)
            decoded["encoded_prev_action"] = torch.rand(
                batch, prev_action_len, tdgu.hparams.hidden_dim
            )
            self.num_calls += 1
            return decoded

    monkeypatch.setattr(tdgu, "forward", MockForward())
    decoded_list = tdgu.greedy_decode(
        TWCmdGenGraphEventStepInput(
            obs_word_ids=torch.randint(tdgu.preprocessor.vocab_size, (batch, obs_len)),
            obs_mask=torch.randint(2, (batch, obs_len)).float(),
            prev_action_word_ids=torch.randint(
                tdgu.preprocessor.vocab_size, (batch, prev_action_len)
            ),
            prev_action_mask=torch.randint(2, (batch, prev_action_len)).float(),
            timestamps=torch.tensor([4.0] * batch),
        ),
        prev_batched_graph,
        max_event_decode_len=max_event_decode_len,
        max_label_decode_len=max_label_decode_len,
        gumbel_greedy_decode=gumbel_greedy_decode,
        gumbel_tau=gumbel_tau,
    )

    assert len(decoded_list) == len(expected_decoded_list)
    for decoded, expected in zip(decoded_list, expected_decoded_list):
        assert decoded["decoded_event_type_ids"].equal(
            expected["decoded_event_type_ids"]
        )
        assert decoded["decoded_event_src_ids"].equal(expected["decoded_event_src_ids"])
        assert decoded["decoded_event_dst_ids"].equal(expected["decoded_event_dst_ids"])
        assert decoded["decoded_event_label_word_ids"].equal(
            expected["decoded_event_label_word_ids"]
        )
        assert decoded["decoded_event_label_mask"].equal(
            expected["decoded_event_label_mask"]
        )
        assert decoded["updated_batched_graph"].batch.equal(
            expected["updated_batched_graph"].batch
        )
        assert decoded["updated_batched_graph"].x.equal(
            expected["updated_batched_graph"].x
        )
        assert decoded["updated_batched_graph"].node_label_mask.equal(
            expected["updated_batched_graph"].node_label_mask
        )
        assert decoded["updated_batched_graph"].node_last_update.equal(
            expected["updated_batched_graph"].node_last_update
        )
        assert decoded["updated_batched_graph"].edge_index.equal(
            expected["updated_batched_graph"].edge_index
        )
        assert decoded["updated_batched_graph"].edge_attr.equal(
            expected["updated_batched_graph"].edge_attr
        )
        assert decoded["updated_batched_graph"].edge_label_mask.equal(
            expected["updated_batched_graph"].edge_label_mask
        )
        assert decoded["updated_batched_graph"].edge_last_update.equal(
            expected["updated_batched_graph"].edge_last_update
        )
        assert decoded["batch_node_embeddings"].equal(expected["batch_node_embeddings"])
        assert decoded["batch_node_mask"].equal(expected["batch_node_mask"])
