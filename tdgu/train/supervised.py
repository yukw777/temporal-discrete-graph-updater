import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import tqdm
import networkx as nx

from typing import Optional, Dict, List, Sequence, Tuple, Any
from torch.optim import AdamW, Optimizer

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer.states import RunningStage
from torch_geometric.data import Batch, Data
from torch.utils.data import DataLoader
from torchmetrics.classification.f_beta import F1Score

from tdgu.metrics import F1, ExactMatch
from tdgu.nn.utils import (
    index_edge_attr,
    masked_log_softmax,
    calculate_node_id_offsets,
    masked_softmax,
)
from tdgu.data import (
    TWCmdGenGraphEventBatch,
    TWCmdGenGraphEventFreeRunDataset,
    TWCmdGenGraphEventGraphicalInput,
    TWCmdGenGraphEventStepInput,
    TWCmdGenGraphEventDataCollator,
)
from tdgu.constants import EVENT_TYPE_ID_MAP
from tdgu.graph import (
    batch_to_data_list,
    data_list_to_batch,
    networkx_to_rdf,
    update_rdf_graph,
)
from tdgu.train import TDGULightningModule


class SupervisedTDGU(TDGULightningModule):
    """
    A LightningModule for supervised training of the temporal discrete graph updater.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.criterion = UncertaintyWeightedLoss()

        self.val_event_type_f1 = F1Score()
        self.val_src_node_f1 = F1Score()
        self.val_dst_node_f1 = F1Score()
        self.val_label_f1 = F1Score()
        self.test_event_type_f1 = F1Score()
        self.test_src_node_f1 = F1Score()
        self.test_dst_node_f1 = F1Score()
        self.test_label_f1 = F1Score()

        self.val_graph_tf_exact_match = ExactMatch()
        self.val_token_tf_exact_match = ExactMatch()
        self.val_graph_tf_f1 = F1()
        self.val_token_tf_f1 = F1()
        self.test_graph_tf_exact_match = ExactMatch()
        self.test_token_tf_exact_match = ExactMatch()
        self.test_graph_tf_f1 = F1()
        self.test_token_tf_f1 = F1()

        self.val_graph_gd_exact_match = ExactMatch()
        self.val_token_gd_exact_match = ExactMatch()
        self.val_graph_gd_f1 = F1()
        self.val_token_gd_f1 = F1()
        self.test_graph_gd_exact_match = ExactMatch()
        self.test_token_gd_exact_match = ExactMatch()
        self.test_graph_gd_f1 = F1()
        self.test_token_gd_f1 = F1()

        self.val_free_run_f1 = F1()
        self.val_free_run_em = ExactMatch()
        self.test_free_run_f1 = F1()
        self.test_free_run_em = ExactMatch()

    def decode_label(
        self, label_word_ids: torch.Tensor, label_mask: torch.Tensor
    ) -> List[str]:
        """
        label_word_ids: (batch, label_len)
        label_mask: (batch, label_len)

        output: [label, ...]
        """
        decoded: List[str] = []
        for word_ids, mask in zip(label_word_ids, label_mask):
            decoded.append(
                " ".join(
                    self.preprocessor.convert_ids_to_tokens(word_ids[mask].tolist())[
                        :-1
                    ]
                )
            )
        return decoded

    def teacher_force(
        self,
        step_input: TWCmdGenGraphEventStepInput,
        graphical_input_seq: Sequence[TWCmdGenGraphEventGraphicalInput],
    ) -> List[Dict[str, Any]]:
        """
        step_input: the current step input
        graphical_input_seq: sequence of graphical inputs
        batch_graph: diagonally stacked batch of current graphs

        output:
        [forward pass output, ...]
        """
        prev_input_event_emb_seq: Optional[torch.Tensor] = None
        prev_input_event_emb_seq_mask: Optional[torch.Tensor] = None
        encoded_obs: Optional[torch.Tensor] = None
        encoded_prev_action: Optional[torch.Tensor] = None
        results_list: List[Dict[str, torch.Tensor]] = []
        for graphical_input in graphical_input_seq:
            results = self(
                graphical_input.tgt_event_type_ids,
                graphical_input.tgt_event_src_ids,
                graphical_input.tgt_event_dst_ids,
                graphical_input.tgt_event_label_word_ids,
                graphical_input.tgt_event_label_mask,
                graphical_input.prev_batched_graph,
                step_input.obs_mask,
                step_input.prev_action_mask,
                step_input.timestamps,
                obs_word_ids=step_input.obs_word_ids,
                prev_action_word_ids=step_input.prev_action_word_ids,
                encoded_obs=encoded_obs,
                encoded_prev_action=encoded_prev_action,
                prev_input_event_emb_seq=prev_input_event_emb_seq,
                prev_input_event_emb_seq_mask=prev_input_event_emb_seq_mask,
                groundtruth_event={
                    "groundtruth_event_type_ids": (
                        graphical_input.groundtruth_event_type_ids
                    ),
                    "groundtruth_event_mask": graphical_input.groundtruth_event_mask,
                    "groundtruth_event_src_ids": (
                        graphical_input.groundtruth_event_src_ids
                    ),
                    "groundtruth_event_src_mask": (
                        graphical_input.groundtruth_event_src_mask
                    ),
                    "groundtruth_event_dst_ids": (
                        graphical_input.groundtruth_event_dst_ids
                    ),
                    "groundtruth_event_dst_mask": (
                        graphical_input.groundtruth_event_dst_mask
                    ),
                    "groundtruth_event_label_tgt_word_ids": (
                        graphical_input.groundtruth_event_label_tgt_word_ids
                    ),
                    "groundtruth_event_label_tgt_mask": (
                        graphical_input.groundtruth_event_label_tgt_mask
                    ),
                },
                max_label_decode_len=self.hparams.max_label_decode_len,  # type: ignore
            )

            # add results to the list
            results_list.append(results)

            # update previous input event embedding sequence
            prev_input_event_emb_seq = results["updated_prev_input_event_emb_seq"]
            prev_input_event_emb_seq_mask = results[
                "updated_prev_input_event_emb_seq_mask"
            ]

            # save the encoded obs and prev action
            encoded_obs = results["encoded_obs"]
            encoded_prev_action = results["encoded_prev_action"]

        return results_list

    def calculate_f1s(
        self,
        event_type_logits: torch.Tensor,
        groundtruth_event_type_ids: torch.Tensor,
        event_src_logits: torch.Tensor,
        groundtruth_event_src_ids: torch.Tensor,
        event_dst_logits: torch.Tensor,
        groundtruth_event_dst_ids: torch.Tensor,
        batch_node_mask: torch.Tensor,
        event_label_logits: torch.Tensor,
        groundtruth_event_label_word_ids: torch.Tensor,
        groundtruth_event_label_word_mask: torch.Tensor,
        groundtruth_event_mask: torch.Tensor,
        groundtruth_event_src_mask: torch.Tensor,
        groundtruth_event_dst_mask: torch.Tensor,
        groundtruth_event_label_mask: torch.Tensor,
        log_prefix: str,
    ) -> None:
        """
        Calculate various F1 scores.

        event_type_logits: (batch, num_event_type)
        groundtruth_event_type_ids: (batch)
        event_src_logits: (batch, num_node)
        groundtruth_event_src_ids: (batch)
        event_dst_logits: (batch, num_node)
        groundtruth_event_dst_ids: (batch)
        batch_node_mask: (batch, num_node)
        event_label_logits: (batch, groundtruth_event_label_len, num_word)
        groundtruth_event_label_word_ids: (batch, groundtruth_event_label_len)
        groundtruth_event_label_word_mask: (batch, groundtruth_event_label_len)
        groundtruth_event_mask: (batch)
        groundtruth_event_src_mask: (batch)
        groundtruth_event_dst_mask: (batch)
        groundtruth_event_label_mask: (batch)
        """
        event_type_f1 = getattr(self, f"{log_prefix}_event_type_f1")
        src_node_f1 = getattr(self, f"{log_prefix}_src_node_f1")
        dst_node_f1 = getattr(self, f"{log_prefix}_dst_node_f1")
        label_f1 = getattr(self, f"{log_prefix}_label_f1")

        event_type_f1.update(
            event_type_logits[groundtruth_event_mask].softmax(dim=1),
            groundtruth_event_type_ids[groundtruth_event_mask],
        )
        if groundtruth_event_src_mask.any():
            src_node_f1.update(
                masked_softmax(
                    event_src_logits[groundtruth_event_src_mask],
                    batch_node_mask[groundtruth_event_src_mask],
                    dim=1,
                ),
                groundtruth_event_src_ids[groundtruth_event_src_mask],
            )
        if groundtruth_event_dst_mask.any():
            dst_node_f1.update(
                masked_softmax(
                    event_dst_logits[groundtruth_event_dst_mask],
                    batch_node_mask[groundtruth_event_dst_mask],
                    dim=1,
                ),
                groundtruth_event_dst_ids[groundtruth_event_dst_mask],
            )
        if groundtruth_event_label_mask.any():
            combined_label_mask = (
                groundtruth_event_label_word_mask
                * groundtruth_event_label_mask.unsqueeze(-1)
            )
            label_f1.update(
                event_label_logits[combined_label_mask].softmax(dim=1),
                groundtruth_event_label_word_ids[combined_label_mask],
            )

    def generate_graph_triples(
        self,
        event_type_ids: torch.Tensor,
        src_ids: torch.Tensor,
        dst_ids: torch.Tensor,
        batch_label_word_ids: torch.Tensor,
        batch_label_mask: torch.Tensor,
        batched_graph: Batch,
    ) -> Tuple[List[str], List[List[str]]]:
        """
        Generate graph triplets based on the given batch of graph events.

        event_type_ids: (batch)
        src_ids: (batch)
        dst_ids: (batch)
        batch_label_word_ids: (batch, label_len)
        batch_label_mask: (batch, label_len)
        batched_graph: batched graph before the given events

        output: (
            cmds: len([commands, ...]) = batch
            tokens: len([[token, ...], ...]) = batch
        )
        """
        node_id_offsets = calculate_node_id_offsets(
            event_type_ids.size(0), batched_graph.batch
        )
        # (batch)
        batch_src_ids = src_ids + node_id_offsets
        # (batch)
        batch_dst_ids = dst_ids + node_id_offsets
        # (batch)

        cmds: List[str] = []
        tokens: List[List[str]] = []
        for event_type_id, src_id, dst_id, label_word_ids, label_mask in zip(
            event_type_ids.tolist(),
            batch_src_ids,
            batch_dst_ids,
            batch_label_word_ids,
            batch_label_mask,
        ):
            if event_type_id in {
                EVENT_TYPE_ID_MAP["edge-add"],
                EVENT_TYPE_ID_MAP["edge-delete"],
            }:
                src_label = self.decode_label(
                    batched_graph.x[src_id].unsqueeze(0),
                    batched_graph.node_label_mask[src_id].unsqueeze(0),
                )[0]
                dst_label = self.decode_label(
                    batched_graph.x[dst_id].unsqueeze(0),
                    batched_graph.node_label_mask[dst_id].unsqueeze(0),
                )[0]
                if event_type_id == EVENT_TYPE_ID_MAP["edge-add"]:
                    cmd = "add"
                    edge_label = self.decode_label(
                        label_word_ids.unsqueeze(0), label_mask.unsqueeze(0)
                    )[0]
                else:
                    cmd = "delete"
                    edge_label_word_ids = index_edge_attr(
                        batched_graph.edge_index,
                        batched_graph.edge_attr,
                        torch.stack([src_id.unsqueeze(0), dst_id.unsqueeze(0)]),
                    )
                    edge_label_mask = index_edge_attr(
                        batched_graph.edge_index,
                        batched_graph.edge_label_mask,
                        torch.stack([src_id.unsqueeze(0), dst_id.unsqueeze(0)]),
                    )
                    edge_label = self.decode_label(
                        edge_label_word_ids, edge_label_mask
                    )[0]
                # in the original dataset, multi-word edge labels are joined by
                # an underscore
                cmd_tokens = [cmd, src_label, dst_label, "_".join(edge_label.split())]
                cmds.append(" , ".join(cmd_tokens))
                tokens.append(cmd_tokens)
            else:
                cmds.append("")
                tokens.append([])
        return cmds, tokens

    def generate_batch_graph_triples_seq(
        self,
        event_type_id_seq: Sequence[torch.Tensor],
        src_id_seq: Sequence[torch.Tensor],
        dst_id_seq: Sequence[torch.Tensor],
        label_word_id_seq: Sequence[torch.Tensor],
        label_mask_seq: Sequence[torch.Tensor],
        batched_graph_seq: Sequence[Batch],
    ) -> Tuple[List[List[str]], List[List[str]]]:
        batch_size = event_type_id_seq[0].size(0)
        # (batch, event_seq_len, cmd_len)
        batch_cmds: List[List[str]] = [[] for _ in range(batch_size)]
        # (batch, event_seq_len, token_len)
        batch_tokens: List[List[str]] = [[] for _ in range(batch_size)]
        for (
            event_type_ids,
            src_ids,
            dst_ids,
            label_word_ids,
            label_mask,
            batched_graph,
        ) in zip(
            event_type_id_seq,
            src_id_seq,
            dst_id_seq,
            label_word_id_seq,
            label_mask_seq,
            batched_graph_seq,
        ):
            for batch_id, (cmd, tokens) in enumerate(
                zip(
                    *self.generate_graph_triples(
                        event_type_ids,
                        src_ids,
                        dst_ids,
                        label_word_ids,
                        label_mask,
                        batched_graph,
                    )
                )
            ):
                if cmd != "":
                    batch_cmds[batch_id].append(cmd)
                if len(tokens) != 0:
                    if len(batch_tokens[batch_id]) != 0:
                        batch_tokens[batch_id].append("<sep>")
                    batch_tokens[batch_id].extend(tokens)
        return batch_cmds, batch_tokens

    @staticmethod
    def generate_batch_groundtruth_graph_triple_tokens(
        groundtruth_cmd_seq: Sequence[Sequence[str]],
    ) -> List[List[str]]:
        batch_groundtruth_tokens: List[List[str]] = []
        for groundtruth_cmds in groundtruth_cmd_seq:
            batch_groundtruth_tokens.append(
                " , <sep> , ".join(groundtruth_cmds).split(" , ")
            )
        return batch_groundtruth_tokens

    def training_step(  # type: ignore
        self, batch: TWCmdGenGraphEventBatch, batch_idx: int
    ) -> torch.Tensor:
        """
        batch: the batch data
        batch_idx: the batch id, unused
        """
        results_list = self.teacher_force(batch.step_input, batch.graphical_input_seq)
        loss = torch.stack(
            [
                self.criterion(
                    results["event_type_logits"],
                    graphical_input.groundtruth_event_type_ids,
                    results["event_src_logits"],
                    graphical_input.groundtruth_event_src_ids,
                    results["event_dst_logits"],
                    graphical_input.groundtruth_event_dst_ids,
                    results["batch_node_mask"],
                    results["event_label_logits"],
                    graphical_input.groundtruth_event_label_groundtruth_word_ids,
                    graphical_input.groundtruth_event_label_tgt_mask,
                    graphical_input.groundtruth_event_mask,
                    graphical_input.groundtruth_event_src_mask,
                    graphical_input.groundtruth_event_dst_mask,
                    graphical_input.groundtruth_event_label_mask,
                )
                for results, graphical_input in zip(
                    results_list, batch.graphical_input_seq
                )
            ]
        ).mean()
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def eval_step(
        self, batch: TWCmdGenGraphEventBatch, log_prefix: str
    ) -> List[Tuple[str, ...]]:
        # [(id, groundtruth commands, teacher-force commands, greedy-decode commands)]
        # id = (game|walkthrough_step|random_step)
        table_data: List[Tuple[str, ...]] = []

        # loss from teacher forcing
        tf_results_list = self.teacher_force(
            batch.step_input, batch.graphical_input_seq
        )
        loss = torch.stack(
            [
                self.criterion(
                    results["event_type_logits"],
                    graphical_input.groundtruth_event_type_ids,
                    results["event_src_logits"],
                    graphical_input.groundtruth_event_src_ids,
                    results["event_dst_logits"],
                    graphical_input.groundtruth_event_dst_ids,
                    results["batch_node_mask"],
                    results["event_label_logits"],
                    graphical_input.groundtruth_event_label_groundtruth_word_ids,
                    graphical_input.groundtruth_event_label_tgt_mask,
                    graphical_input.groundtruth_event_mask,
                    graphical_input.groundtruth_event_src_mask,
                    graphical_input.groundtruth_event_dst_mask,
                    graphical_input.groundtruth_event_label_mask,
                )
                for results, graphical_input in zip(
                    tf_results_list, batch.graphical_input_seq
                )
            ]
        ).mean()
        # TODO: Remove batch_size when
        # https://github.com/PyTorchLightning/pytorch-lightning/pull/12573
        # is released.
        self.log(log_prefix + "_loss", loss, batch_size=len(batch.ids))

        # log classification F1s from teacher forcing
        for results, graphical_input in zip(tf_results_list, batch.graphical_input_seq):
            self.calculate_f1s(
                results["event_type_logits"],
                graphical_input.groundtruth_event_type_ids,
                results["event_src_logits"],
                graphical_input.groundtruth_event_src_ids,
                results["event_dst_logits"],
                graphical_input.groundtruth_event_dst_ids,
                results["batch_node_mask"],
                results["event_label_logits"],
                graphical_input.groundtruth_event_label_groundtruth_word_ids,
                graphical_input.groundtruth_event_label_tgt_mask,
                graphical_input.groundtruth_event_mask,
                graphical_input.groundtruth_event_src_mask,
                graphical_input.groundtruth_event_dst_mask,
                graphical_input.groundtruth_event_label_mask,
                log_prefix,
            )
        self.log(
            log_prefix + "_event_type_f1", getattr(self, f"{log_prefix}_event_type_f1")
        )
        self.log(
            log_prefix + "_src_node_f1", getattr(self, f"{log_prefix}_src_node_f1")
        )
        self.log(
            log_prefix + "_dst_node_f1", getattr(self, f"{log_prefix}_dst_node_f1")
        )
        self.log(log_prefix + "_label_f1", getattr(self, f"{log_prefix}_label_f1"))

        # calculate graph tuples from teacher forcing graph events
        tf_event_type_id_seq: List[torch.Tensor] = []
        tf_src_id_seq: List[torch.Tensor] = []
        tf_dst_id_seq: List[torch.Tensor] = []
        tf_label_word_id_seq: List[torch.Tensor] = []
        for results, graphical_input in zip(tf_results_list, batch.graphical_input_seq):
            # filter out pad events
            unfiltered_event_type_ids = (
                results["event_type_logits"].argmax(dim=1)
                * graphical_input.groundtruth_event_mask
            )
            # handle source/destination logits for empty graphs by setting them to zeros
            unfiltered_event_src_ids = (
                masked_softmax(
                    results["event_src_logits"], results["batch_node_mask"], dim=1
                ).argmax(dim=1)
                if results["event_src_logits"].size(1) > 0
                else torch.zeros(
                    results["event_src_logits"].size(0),
                    dtype=torch.long,
                    device=self.device,
                )
            )
            unfiltered_event_dst_ids = (
                masked_softmax(
                    results["event_dst_logits"], results["batch_node_mask"], dim=1
                ).argmax(dim=1)
                if results["event_dst_logits"].size(1) > 0
                else torch.zeros(
                    results["event_dst_logits"].size(0),
                    dtype=torch.long,
                    device=self.device,
                )
            )
            invalid_event_mask = self.filter_invalid_events(
                unfiltered_event_type_ids,
                unfiltered_event_src_ids,
                unfiltered_event_dst_ids,
                results["updated_batched_graph"].batch,
                results["updated_batched_graph"].edge_index,
            )
            tf_event_type_id_seq.append(
                unfiltered_event_type_ids.masked_fill(
                    invalid_event_mask, EVENT_TYPE_ID_MAP["pad"]
                )
            )
            tf_src_id_seq.append(
                unfiltered_event_src_ids.masked_fill(invalid_event_mask, 0)
            )
            tf_dst_id_seq.append(
                unfiltered_event_dst_ids.masked_fill(invalid_event_mask, 0)
            )
            tf_label_word_id_seq.append(results["event_label_logits"].argmax(dim=2))

        batch_tf_cmds, batch_tf_tokens = self.generate_batch_graph_triples_seq(
            tf_event_type_id_seq,
            tf_src_id_seq,
            tf_dst_id_seq,
            tf_label_word_id_seq,
            [
                graphical_input.groundtruth_event_label_tgt_mask
                for graphical_input in batch.graphical_input_seq
            ],
            [results["updated_batched_graph"] for results in tf_results_list],
        )

        # collect groundtruth command tokens
        batch_groundtruth_tokens = self.generate_batch_groundtruth_graph_triple_tokens(
            batch.graph_commands
        )

        # log teacher force graph based metrics
        graph_tf_exact_match = getattr(self, f"{log_prefix}_graph_tf_exact_match")
        graph_tf_exact_match.update(batch_tf_cmds, batch.graph_commands)
        self.log(log_prefix + "_graph_tf_em", graph_tf_exact_match)
        graph_tf_f1 = getattr(self, f"{log_prefix}_graph_tf_f1")
        graph_tf_f1.update(batch_tf_cmds, batch.graph_commands)
        self.log(log_prefix + "_graph_tf_f1", graph_tf_f1)

        # log teacher force token based metrics
        token_tf_exact_match = getattr(self, f"{log_prefix}_token_tf_exact_match")
        token_tf_exact_match.update(batch_tf_tokens, batch_groundtruth_tokens)
        self.log(log_prefix + "_token_tf_em", token_tf_exact_match)
        token_tf_f1 = getattr(self, f"{log_prefix}_token_tf_f1")
        token_tf_f1.update(batch_tf_tokens, batch_groundtruth_tokens)
        self.log(log_prefix + "_token_tf_f1", token_tf_f1)

        # greedy decoding
        gd_results_list = self.greedy_decode(
            batch.step_input,
            batch.initial_batched_graph,
            max_event_decode_len=self.hparams.max_event_decode_len,  # type: ignore
        )

        # calculate graph tuples from greedy decoded graph events
        batch_gd_cmds, batch_gd_tokens = self.generate_batch_graph_triples_seq(
            [results["decoded_event_type_ids"] for results in gd_results_list],
            [results["decoded_event_src_ids"] for results in gd_results_list],
            [results["decoded_event_dst_ids"] for results in gd_results_list],
            [results["decoded_event_label_word_ids"] for results in gd_results_list],
            [results["decoded_event_label_mask"] for results in gd_results_list],
            [results["updated_batched_graph"] for results in gd_results_list],
        )

        # log greedy decode graph based metrics
        graph_gd_exact_match = getattr(self, log_prefix + "_graph_gd_exact_match")
        graph_gd_exact_match.update(batch_gd_cmds, batch.graph_commands)
        self.log(log_prefix + "_graph_gd_em", graph_gd_exact_match)
        graph_gd_f1 = getattr(self, log_prefix + "_graph_gd_f1")
        graph_gd_f1.update(batch_gd_cmds, batch.graph_commands)
        self.log(log_prefix + "_graph_gd_f1", graph_gd_f1)

        # log greedy decode token based metrics
        token_gd_exact_match = getattr(self, log_prefix + "_token_gd_exact_match")
        token_gd_exact_match.update(batch_gd_tokens, batch_groundtruth_tokens)
        self.log(log_prefix + "_token_gd_em", token_gd_exact_match)
        token_gd_f1 = getattr(self, log_prefix + "_token_gd_f1")
        token_gd_f1.update(batch_gd_tokens, batch_groundtruth_tokens)
        self.log(log_prefix + "_token_gd_f1", token_gd_f1)

        # collect graph triple table data
        table_data.extend(
            self.generate_predict_table_rows(
                batch.ids, batch.graph_commands, batch_tf_cmds, batch_gd_cmds
            )
        )

        return table_data

    def data_to_networkx(self, data: Data) -> nx.DiGraph:
        """
        Turn torch_geometric.Data into a networkx graph.

        There is a bug in to_networkx() where it turns an attribute that is a list
        with one element into a scalar, so we just manually construct a networkx graph.
        """
        graph = nx.DiGraph()
        graph.add_nodes_from(
            [
                (
                    nid,
                    {
                        "node_last_update": node_last_update,
                        "label": self.decode_label(
                            node_label_word_ids.unsqueeze(0),
                            node_label_mask.unsqueeze(0),
                        )[0],
                    },
                )
                for nid, (
                    node_label_word_ids,
                    node_label_mask,
                    node_last_update,
                ) in enumerate(
                    zip(data.x, data.node_label_mask, data.node_last_update.tolist())
                )
            ]
        )
        graph.add_edges_from(
            [
                (
                    src_id,
                    dst_id,
                    {
                        "edge_last_update": edge_last_update,
                        "label": self.decode_label(
                            edge_label_word_ids.unsqueeze(0),
                            edge_label_mask.unsqueeze(0),
                        )[0],
                    },
                )
                for (
                    src_id,
                    dst_id,
                ), edge_label_word_ids, edge_label_mask, edge_last_update in zip(
                    data.edge_index.t().tolist(),
                    data.edge_attr,
                    data.edge_label_mask,
                    data.edge_last_update.tolist(),
                )
            ]
        )
        return graph

    def eval_free_run(
        self,
        dataset: TWCmdGenGraphEventFreeRunDataset,
        dataloader: DataLoader,
        collator: TWCmdGenGraphEventDataCollator,
        total: Optional[int] = None,
    ) -> Tuple[List[List[str]], List[List[str]]]:
        if total is None:
            is_sanity_checking = (
                self.trainer.state.stage == RunningStage.SANITY_CHECKING  # type: ignore
            )
            if is_sanity_checking:
                total = self.trainer.num_sanity_val_steps  # type: ignore
            elif self.trainer.state.stage == RunningStage.VALIDATING:  # type: ignore
                if isinstance(self.trainer.limit_val_batches, float):  # type: ignore
                    total = int(
                        self.trainer.limit_val_batches * len(dataset)  # type: ignore
                    )
                elif isinstance(self.trainer.limit_val_batches, int):  # type: ignore
                    total = self.trainer.limit_val_batches  # type: ignore
                else:
                    total = len(dataset)
            elif self.trainer.state.stage == RunningStage.TESTING:  # type: ignore
                if isinstance(self.trainer.limit_test_batches, float):  # type: ignore
                    total = int(
                        self.trainer.limit_test_batches * len(dataset)  # type: ignore
                    )
                elif isinstance(self.trainer.limit_test_batches, int):  # type: ignore
                    total = self.trainer.limit_test_batches  # type: ignore
                else:
                    total = len(dataset)

        game_id_to_step_data_graph: Dict[int, Tuple[Dict[str, Any], Data]] = {}
        generated_rdfs_list: List[List[str]] = []
        groundtruth_rdfs_list: List[List[str]] = []
        with tqdm.tqdm(desc="Free Run", total=total) as pbar:
            for batch in dataloader:
                # finished games are the ones that were in game_id_to_graph, but are not
                # part of the new batch
                for finished_game_id in game_id_to_step_data_graph.keys() - {
                    game_id for game_id, _ in batch
                }:
                    step_data, graph = game_id_to_step_data_graph.pop(finished_game_id)
                    generated_rdfs = networkx_to_rdf(
                        self.data_to_networkx(graph),
                        allow_objs_with_same_label=(
                            self.hparams.allow_objs_with_same_label  # type: ignore
                        ),
                    )
                    groundtruth_rdfs = update_rdf_graph(
                        set(step_data["previous_graph_seen"]),
                        step_data["target_commands"],
                    )
                    generated_rdfs_list.append(list(generated_rdfs))
                    groundtruth_rdfs_list.append(list(groundtruth_rdfs))
                    pbar.update()
                    if pbar.n > total:
                        return generated_rdfs_list, groundtruth_rdfs_list

                # new games are the ones that were not in game_id_to_graph, but are now
                # part of the new batch.
                # due to Python's dictionary ordering (insertion order), new games are
                # added always to the end.
                for game_id, step_data in batch:
                    if game_id in game_id_to_step_data_graph:
                        _, graph = game_id_to_step_data_graph[game_id]
                        game_id_to_step_data_graph[game_id] = (step_data, graph)
                    else:
                        game_id_to_step_data_graph[game_id] = (
                            step_data,
                            Data(
                                x=torch.empty(0, 0, dtype=torch.long),
                                node_label_mask=torch.empty(0, 0).bool(),
                                node_last_update=torch.empty(0, 2, dtype=torch.long),
                                edge_index=torch.empty(2, 0, dtype=torch.long),
                                edge_attr=torch.empty(0, 0).long(),
                                edge_label_mask=torch.empty(0, 0).bool(),
                                edge_last_update=torch.empty(0, 2, dtype=torch.long),
                            ).to(self.device),
                        )

                # sanity check
                assert [game_id for game_id, _ in batch] == [
                    game_id for game_id in game_id_to_step_data_graph
                ]

                # construct a batch
                batched_obs: List[str] = []
                batched_prev_actions: List[str] = []
                batched_timestamps: List[int] = []
                graph_list: List[Data] = []
                for game_id, (step_data, graph) in game_id_to_step_data_graph.items():
                    batched_obs.append(step_data["observation"])
                    batched_prev_actions.append(step_data["previous_action"])
                    batched_timestamps.append(step_data["timestamp"])
                    graph_list.append(graph)

                # greedy decode
                results_list = self.greedy_decode(
                    collator.collate_step_inputs(  # type: ignore
                        batched_obs, batched_prev_actions, batched_timestamps
                    ).to(self.device),
                    data_list_to_batch(graph_list),
                )

                # update graphs in game_id_to_step_data_graph
                for (game_id, (step_data, _)), updated_graph in zip(
                    game_id_to_step_data_graph.items(),
                    batch_to_data_list(
                        results_list[-1]["updated_batched_graph"], len(batched_obs)
                    ),
                ):
                    game_id_to_step_data_graph[game_id] = (step_data, updated_graph)
        return generated_rdfs_list, groundtruth_rdfs_list

    def validation_step(  # type: ignore
        self, batch: TWCmdGenGraphEventBatch, batch_idx: int
    ) -> List[Tuple[str, ...]]:
        return self.eval_step(batch, "val")

    def test_step(  # type: ignore
        self, batch: TWCmdGenGraphEventBatch, batch_idx: int
    ) -> List[Tuple[str, ...]]:
        return self.eval_step(batch, "test")

    def wandb_log_gen_obs(
        self, outputs: List[List[Tuple[str, str, str, str]]], table_title: str
    ) -> None:
        eval_table_artifact = wandb.Artifact(
            table_title + f"_{self.logger.experiment.id}", "predictions"  # type: ignore
        )
        eval_table = wandb.Table(
            columns=["id", "truth", "tf", "gd"],
            data=[item for sublist in outputs for item in sublist],
        )
        eval_table_artifact.add(eval_table, "predictions")
        self.logger.experiment.log_artifact(eval_table_artifact)  # type: ignore

    def validation_epoch_end(  # type: ignore
        self,
        outputs: List[List[Tuple[str, str, str, str]]],
    ) -> None:
        generated_rdfs_list, groundtruth_rdfs_list = self.eval_free_run(
            self.trainer.datamodule.valid_free_run,  # type: ignore
            self.trainer.datamodule.val_free_run_dataloader(),  # type: ignore
            self.trainer.datamodule.collator,  # type: ignore
        )
        self.val_free_run_f1.update(generated_rdfs_list, groundtruth_rdfs_list)
        self.log("val_free_run_f1", self.val_free_run_f1, prog_bar=True)
        self.val_free_run_em.update(generated_rdfs_list, groundtruth_rdfs_list)
        self.log("val_free_run_em", self.val_free_run_em)
        if isinstance(self.logger, WandbLogger):
            self.wandb_log_gen_obs(outputs, "val_gen_graph_triples")

    def test_epoch_end(  # type: ignore
        self,
        outputs: List[List[Tuple[str, str, str, str]]],
    ) -> None:
        generated_rdfs_list, groundtruth_rdfs_list = self.eval_free_run(
            self.trainer.datamodule.test_free_run,  # type: ignore
            self.trainer.datamodule.test_free_run_dataloader(),  # type: ignore
            self.trainer.datamodule.collator,  # type: ignore
        )
        self.test_free_run_f1.update(generated_rdfs_list, groundtruth_rdfs_list)
        self.log("test_free_run_f1", self.test_free_run_f1)
        self.test_free_run_em.update(generated_rdfs_list, groundtruth_rdfs_list)
        self.log("test_free_run_em", self.test_free_run_em)
        if isinstance(self.logger, WandbLogger):
            self.wandb_log_gen_obs(outputs, "test_gen_graph_triples")

    def configure_optimizers(self) -> Optimizer:
        return AdamW(self.parameters(), lr=self.hparams.learning_rate)  # type: ignore

    def greedy_decode(
        self,
        step_input: TWCmdGenGraphEventStepInput,
        prev_batched_graph: Batch,
        max_event_decode_len: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        step_input: the current step input
        prev_batch_graph: diagonally stacked batch of current graphs
        max_event_decode_len: max length of decoded event sequence

        output:
        len([{
            decoded_event_type_ids: (batch)
            decoded_event_src_ids: (batch)
            decoded_event_dst_ids: (batch)
            decoded_event_label_word_ids: (batch, decoded_label_len)
            decoded_event_label_mask: (batch, decoded_label_len)
            updated_batched_graph: diagonally stacked batch of updated graphs.
                these are the graphs used to decode the graph events above
            self_attn_weights:
                len([(batch, 1, input_seq_len), ...]) == num_decoder_block,
            obs_graph_attn_weights:
                len([(batch, 1, obs_len), ...]) == num_decoder_block,
            prev_action_graph_attn_weights:
                len([(batch, 1, prev_action_len), ...]) == num_decoder_block,
            graph_obs_attn_weights:
                len([(batch, 1, num_node), ...]) == num_decoder_block,
            graph_prev_action_attn_weights:
                len([(batch, 1, num_node), ...]) == num_decoder_block,
        }, ...]) == decode_len <= max_event_decode_len
        """
        # initialize the initial inputs
        batch_size = step_input.obs_word_ids.size(0)
        decoded_event_type_ids = torch.tensor(
            [EVENT_TYPE_ID_MAP["start"]] * batch_size, device=self.device
        )
        decoded_src_ids = torch.zeros(batch_size, device=self.device, dtype=torch.long)
        decoded_dst_ids = torch.zeros(batch_size, device=self.device, dtype=torch.long)
        decoded_label_word_ids = torch.empty(
            batch_size, 0, device=self.device, dtype=torch.long
        )
        decoded_label_mask = torch.empty(
            batch_size, 0, device=self.device, dtype=torch.bool
        )

        end_event_mask = torch.tensor([False] * batch_size, device=self.device)
        # (batch)

        prev_input_event_emb_seq: Optional[torch.Tensor] = None
        prev_input_event_emb_seq_mask: Optional[torch.Tensor] = None
        encoded_obs: Optional[torch.Tensor] = None
        encoded_prev_action: Optional[torch.Tensor] = None
        results_list: List[Dict[str, Any]] = []
        for _ in range(max_event_decode_len):
            results = self(
                decoded_event_type_ids,
                decoded_src_ids,
                decoded_dst_ids,
                decoded_label_word_ids,
                decoded_label_mask,
                prev_batched_graph,
                step_input.obs_mask,
                step_input.prev_action_mask,
                step_input.timestamps,
                obs_word_ids=step_input.obs_word_ids,
                prev_action_word_ids=step_input.prev_action_word_ids,
                encoded_obs=encoded_obs,
                encoded_prev_action=encoded_prev_action,
                prev_input_event_emb_seq=prev_input_event_emb_seq,
                prev_input_event_emb_seq_mask=prev_input_event_emb_seq_mask,
            )

            # process the decoded result for the next iteration
            decoded_event_type_ids = (
                results["event_type_logits"]
                .argmax(dim=1)
                .masked_fill(end_event_mask, EVENT_TYPE_ID_MAP["pad"])
            )
            # (batch)

            if results["event_src_logits"].size(1) == 0:
                decoded_src_ids = torch.zeros(
                    batch_size, dtype=torch.long, device=self.device
                )
            else:
                decoded_src_ids = masked_softmax(
                    results["event_src_logits"], results["batch_node_mask"], dim=1
                ).argmax(dim=1)
            # (batch)
            if results["event_dst_logits"].size(1) == 0:
                decoded_dst_ids = torch.zeros(
                    batch_size, dtype=torch.long, device=self.device
                )
            else:
                decoded_dst_ids = masked_softmax(
                    results["event_dst_logits"], results["batch_node_mask"], dim=1
                ).argmax(dim=1)
            # (batch)
            decoded_label_word_ids = results["decoded_event_label_word_ids"]
            decoded_label_mask = results["decoded_event_label_mask"]

            # filter out invalid decoded events
            invalid_event_mask = self.filter_invalid_events(
                decoded_event_type_ids,
                decoded_src_ids,
                decoded_dst_ids,
                results["updated_batched_graph"].batch,
                results["updated_batched_graph"].edge_index,
            )
            # (batch)
            decoded_event_type_ids = decoded_event_type_ids.masked_fill(
                invalid_event_mask, EVENT_TYPE_ID_MAP["pad"]
            )

            # collect the results
            results_list.append(
                {
                    "decoded_event_type_ids": decoded_event_type_ids,
                    "decoded_event_src_ids": decoded_src_ids,
                    "decoded_event_dst_ids": decoded_dst_ids,
                    "decoded_event_label_word_ids": decoded_label_word_ids,
                    "decoded_event_label_mask": decoded_label_mask,
                    "updated_batched_graph": results["updated_batched_graph"],
                    "self_attn_weights": results["self_attn_weights"],
                    "obs_graph_attn_weights": results["obs_graph_attn_weights"],
                    "prev_action_graph_attn_weights": results[
                        "prev_action_graph_attn_weights"
                    ],
                    "graph_obs_attn_weights": results["graph_obs_attn_weights"],
                    "graph_prev_action_attn_weights": results[
                        "graph_prev_action_attn_weights"
                    ],
                }
            )

            # update the batched graph
            prev_batched_graph = results["updated_batched_graph"]

            # update previous input event embedding sequence
            prev_input_event_emb_seq = results["updated_prev_input_event_emb_seq"]
            prev_input_event_emb_seq_mask = results[
                "updated_prev_input_event_emb_seq_mask"
            ]

            # save update the encoded observation and previous action
            encoded_obs = results["encoded_obs"]
            encoded_prev_action = results["encoded_prev_action"]

            # update end_event_mask
            end_event_mask = end_event_mask.logical_or(
                decoded_event_type_ids == EVENT_TYPE_ID_MAP["end"]
            )

            # if everything in the batch is done, break
            if end_event_mask.all():
                break

        return results_list

    @staticmethod
    def filter_invalid_events(
        event_type_ids: torch.Tensor,
        src_ids: torch.Tensor,
        dst_ids: torch.Tensor,
        batch: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return a mask for invalid events. False for valid events and
        True for invalid events.

        node-add: all are valid
        node-delete:
            - nodes should exist
            - nodes cannot have edges
        edge-add:
            - nodes should exist
        edge-delete:
            - nodes should exist

        event_type_ids: (batch)
        src_ids: (batch)
        dst_ids: (batch)
        batch: (num_node)
        edge_index: (2, num_edge)

        output: invalid event mask (batch)
        """
        batch_size = event_type_ids.size(0)
        batch_bincount = batch.bincount()
        subgraph_num_node = F.pad(
            batch_bincount, (0, batch_size - batch_bincount.size(0))
        )
        # (batch)

        invalid_src_mask = src_ids >= subgraph_num_node
        # (batch)
        invalid_dst_mask = dst_ids >= subgraph_num_node
        # (batch)

        # node-delete
        node_id_offsets = calculate_node_id_offsets(event_type_ids.size(0), batch)
        # (batch)
        batch_src_ids = src_ids + node_id_offsets
        # (batch)
        nodes_with_edges = torch.any(
            batch_src_ids.unsqueeze(-1) == edge_index.flatten(), dim=1
        )
        # (batch)
        invalid_node_delete_event_mask = invalid_src_mask.logical_or(
            nodes_with_edges
        ).logical_and(event_type_ids == EVENT_TYPE_ID_MAP["node-delete"])
        # (batch)

        invalid_edge_mask = invalid_src_mask.logical_or(invalid_dst_mask)
        # (batch)
        invalid_edge_add_event_mask = invalid_edge_mask.logical_and(
            event_type_ids == EVENT_TYPE_ID_MAP["edge-add"]
        )
        # (batch)

        # if the edge doesn't exist, we still output it as was done
        # in the original GATA paper
        invalid_edge_delete_event_mask = invalid_edge_mask.logical_and(
            event_type_ids == EVENT_TYPE_ID_MAP["edge-delete"]
        )
        # (batch)

        return invalid_node_delete_event_mask.logical_or(
            invalid_edge_add_event_mask
        ).logical_or(invalid_edge_delete_event_mask)
        # (batch)

    @staticmethod
    def generate_predict_table_rows(
        ids: Sequence[Tuple[str, int, int]], *args: Sequence[Sequence[str]]
    ) -> List[Tuple[str, ...]]:
        """
        Generate rows for the prediction table.

        ids: len([(game, walkthrough_step, random_step), ...]) = batch
        args: various commands of shape (batch, event_seq_len)

        output: [
            ('game|walkthrough_step|random_step', groundtruth_cmd, tf_cmd, gd_cmd),
            ...
        ]
        """
        return list(
            zip(
                ["|".join(map(str, step_id)) for step_id in ids],
                *[[" | ".join(cmds) for cmds in batch_cmds] for batch_cmds in args],
            )
        )


class UncertaintyWeightedLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ce_criterion = nn.CrossEntropyLoss(reduction="none")
        self.nll_criterion = nn.NLLLoss(reduction="none")
        # log variance for event type, source node, destination node and label
        # classification tasks
        self.log_var = nn.parameter.Parameter(torch.zeros(4))

    def forward(
        self,
        event_type_logits: torch.Tensor,
        groundtruth_event_type_ids: torch.Tensor,
        event_src_logits: torch.Tensor,
        groundtruth_event_src_ids: torch.Tensor,
        event_dst_logits: torch.Tensor,
        groundtruth_event_dst_ids: torch.Tensor,
        batch_node_mask: torch.Tensor,
        event_label_logits: torch.Tensor,
        groundtruth_event_label_word_ids: torch.Tensor,
        groundtruth_event_label_word_mask: torch.Tensor,
        groundtruth_event_mask: torch.Tensor,
        groundtruth_event_src_mask: torch.Tensor,
        groundtruth_event_dst_mask: torch.Tensor,
        groundtruth_event_label_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the total loss using the weighting strategy from Kendall, et al. 2018.
        with a small modification from Liebel, et al. 2018.

        event_type_logits: (batch, num_event_type)
        groundtruth_event_type_ids: (batch)
        event_src_logits: (batch, num_node)
        groundtruth_event_src_ids: (batch)
        event_dst_logits: (batch, num_node)
        groundtruth_event_dst_ids: (batch)
        batch_node_mask: (batch, num_node)
        event_label_logits: (batch, groundtruth_event_label_len, num_word)
        groundtruth_event_label_word_ids: (batch, groundtruth_event_label_len)
        groundtruth_event_label_word_mask: (batch, groundtruth_event_label_len)
        groundtruth_event_mask: (batch)
        groundtruth_event_src_mask: (batch)
        groundtruth_event_dst_mask: (batch)
        groundtruth_event_label_mask: (batch)

        output: (batch)
        """
        # event type loss
        event_type_loss = (
            self.ce_criterion(event_type_logits, groundtruth_event_type_ids)
            * groundtruth_event_mask
        )
        # (batch)
        src_node_loss = torch.zeros_like(event_type_loss)
        # (batch)
        if groundtruth_event_src_mask.any():
            # source node loss
            src_node_loss += (
                self.nll_criterion(
                    masked_log_softmax(event_src_logits, batch_node_mask, dim=1),
                    groundtruth_event_src_ids,
                )
                * groundtruth_event_src_mask
            )
        dst_node_loss = torch.zeros_like(event_type_loss)
        # (batch)
        if groundtruth_event_dst_mask.any():
            # destination node loss
            dst_node_loss += (
                self.nll_criterion(
                    masked_log_softmax(event_dst_logits, batch_node_mask, dim=1),
                    groundtruth_event_dst_ids,
                )
                * groundtruth_event_dst_mask
            )
        label_loss = torch.zeros_like(event_type_loss)
        # (batch)
        if groundtruth_event_label_mask.any():
            # label loss
            # first calculate the combined label mask
            combined_label_mask = (
                groundtruth_event_label_word_mask
                * groundtruth_event_label_mask.unsqueeze(-1)
            )
            # (batch, groundtruth_event_label_len)
            label_loss += torch.sum(
                self.ce_criterion(
                    event_label_logits.flatten(end_dim=1),
                    groundtruth_event_label_word_ids.flatten(),
                ).view(combined_label_mask.size(0), -1)
                * combined_label_mask,
                dim=1,
            )

        # calculate the total loss
        precision = torch.exp(-self.log_var)
        # (4)
        stacked_losses = torch.stack(
            [event_type_loss, src_node_loss, dst_node_loss, label_loss]
        ).t()
        # (batch, 4)
        regularizer = torch.log1p(torch.exp(self.log_var))
        # (4)
        return torch.sum(stacked_losses * precision + regularizer, dim=1)
        # (batch)
