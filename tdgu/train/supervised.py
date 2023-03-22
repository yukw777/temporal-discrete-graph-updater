from collections.abc import Sequence
from typing import Any

import networkx as nx
import torch
import torch.nn as nn
import tqdm
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer.states import RunningStage
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data
from torchmetrics.classification import MulticlassF1Score

from tdgu.constants import EVENT_TYPE_ID_MAP, EVENT_TYPES
from tdgu.data import (
    TWCmdGenGraphEventBatch,
    TWCmdGenGraphEventFreeRunDataset,
    TWCmdGenGraphEventGraphicalInput,
    TWCmdGenGraphEventStepInput,
    collate_step_inputs,
)
from tdgu.graph import (
    batch_to_data_list,
    data_list_to_batch,
    networkx_to_rdf,
    update_rdf_graph,
)
from tdgu.metrics import F1, DynamicGraphNodeF1, ExactMatch
from tdgu.nn.graph_updater import TemporalDiscreteGraphUpdater
from tdgu.nn.utils import (
    calculate_node_id_offsets,
    index_edge_attr,
    masked_log_softmax,
    masked_softmax,
)
from tdgu.preprocessor import Preprocessor


class SupervisedTDGU(TemporalDiscreteGraphUpdater):
    """A LightningModule for supervised training of the temporal discrete graph
    updater."""

    def __init__(
        self,
        preprocessor: Preprocessor,
        max_event_decode_len: int,
        max_label_decode_len: int,
        learning_rate: float,
        allow_objs_with_same_label: bool,
        **kwargs,
    ) -> None:
        super().__init__(
            label_head_bos_token_id=preprocessor.bos_token_id,
            label_head_eos_token_id=preprocessor.eos_token_id,
            label_head_pad_token_id=preprocessor.pad_token_id,
            vocab_size=preprocessor.vocab_size,
            **kwargs,
        )
        self.preprocessor = preprocessor
        self.max_event_decode_len = max_event_decode_len
        self.max_label_decode_len = max_label_decode_len
        self.learning_rate = learning_rate
        self.allow_objs_with_same_label = allow_objs_with_same_label

        self.criterion = UncertaintyWeightedLoss()

        self.val_event_type_f1 = MulticlassF1Score(len(EVENT_TYPES), average="micro")
        self.val_src_node_f1 = DynamicGraphNodeF1()
        self.val_dst_node_f1 = DynamicGraphNodeF1()
        self.val_label_f1 = MulticlassF1Score(self.vocab_size, average="micro")
        self.test_event_type_f1 = MulticlassF1Score(len(EVENT_TYPES), average="micro")
        self.test_src_node_f1 = DynamicGraphNodeF1()
        self.test_dst_node_f1 = DynamicGraphNodeF1()
        self.test_label_f1 = MulticlassF1Score(self.vocab_size, average="micro")

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

        self.validation_step_outputs: list[list[tuple[str, ...]]] = []
        self.test_step_outputs: list[list[tuple[str, ...]]] = []

    def teacher_force(
        self,
        step_input: TWCmdGenGraphEventStepInput,
        graphical_input_seq: Sequence[TWCmdGenGraphEventGraphicalInput],
    ) -> list[dict[str, Any]]:
        """
        step_input: the current step input
        graphical_input_seq: sequence of graphical inputs
        batch_graph: diagonally stacked batch of current graphs

        output:
        [forward pass output, ...]
        """
        prev_input_event_emb_seq: torch.Tensor | None = None
        prev_input_event_emb_seq_mask: torch.Tensor | None = None
        encoded_obs: torch.Tensor | None = None
        encoded_prev_action: torch.Tensor | None = None
        results_list: list[dict[str, torch.Tensor]] = []
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
                max_label_decode_len=self.max_label_decode_len,
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
        """Calculate various F1 scores.

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
        event_type_f1: MulticlassF1Score = getattr(self, f"{log_prefix}_event_type_f1")
        src_node_f1: DynamicGraphNodeF1 = getattr(self, f"{log_prefix}_src_node_f1")
        dst_node_f1: DynamicGraphNodeF1 = getattr(self, f"{log_prefix}_dst_node_f1")
        label_f1: MulticlassF1Score = getattr(self, f"{log_prefix}_label_f1")

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
    ) -> tuple[list[str], list[list[str]]]:
        """Generate graph triplets based on the given batch of graph events.

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

        cmds: list[str] = []
        tokens: list[list[str]] = []
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
                src_label = self.preprocessor.batch_decode(
                    batched_graph.x[src_id].unsqueeze(0),
                    batched_graph.node_label_mask[src_id].unsqueeze(0),
                    skip_special_tokens=True,
                )[0]
                dst_label = self.preprocessor.batch_decode(
                    batched_graph.x[dst_id].unsqueeze(0),
                    batched_graph.node_label_mask[dst_id].unsqueeze(0),
                    skip_special_tokens=True,
                )[0]
                if event_type_id == EVENT_TYPE_ID_MAP["edge-add"]:
                    cmd = "add"
                    edge_label = self.preprocessor.batch_decode(
                        label_word_ids.unsqueeze(0),
                        label_mask.unsqueeze(0),
                        skip_special_tokens=True,
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
                    edge_label = self.preprocessor.batch_decode(
                        edge_label_word_ids, edge_label_mask, skip_special_tokens=True
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
    ) -> tuple[list[list[str]], list[list[str]]]:
        batch_size = event_type_id_seq[0].size(0)
        # (batch, event_seq_len, cmd_len)
        batch_cmds: list[list[str]] = [[] for _ in range(batch_size)]
        # (batch, event_seq_len, token_len)
        batch_tokens: list[list[str]] = [[] for _ in range(batch_size)]
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
    ) -> list[list[str]]:
        batch_groundtruth_tokens: list[list[str]] = []
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
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def eval_step(
        self, batch: TWCmdGenGraphEventBatch, log_prefix: str
    ) -> list[tuple[str, ...]]:
        # [(id, groundtruth commands, teacher-force commands, greedy-decode commands)]
        # id = (game|walkthrough_step|random_step)
        table_data: list[tuple[str, ...]] = []

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
        self.log(log_prefix + "_loss", loss, batch_size=len(batch.ids), sync_dist=True)

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
        tf_event_type_id_seq: list[torch.Tensor] = []
        tf_src_id_seq: list[torch.Tensor] = []
        tf_dst_id_seq: list[torch.Tensor] = []
        tf_label_word_id_seq: list[torch.Tensor] = []
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
                else torch.zeros(  # type: ignore
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
                else torch.zeros(  # type: ignore
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
            max_event_decode_len=self.max_event_decode_len,
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
        """Turn torch_geometric.Data into a networkx graph.

        There is a bug in to_networkx() where it turns an attribute that
        is a list with one element into a scalar, so we just manually
        construct a networkx graph.
        """
        graph = nx.DiGraph()
        graph.add_nodes_from(
            [
                (
                    nid,
                    {
                        "node_last_update": node_last_update,
                        "label": self.preprocessor.batch_decode(
                            node_label_word_ids.unsqueeze(0),
                            node_label_mask.unsqueeze(0),
                            skip_special_tokens=True,
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
                        "label": self.preprocessor.batch_decode(
                            edge_label_word_ids.unsqueeze(0),
                            edge_label_mask.unsqueeze(0),
                            skip_special_tokens=True,
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
        total: int | None = None,
    ) -> tuple[list[list[str]], list[list[str]]]:
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

        game_id_to_step_data_graph: dict[int, tuple[dict[str, Any], Data]] = {}
        generated_rdfs_list: list[list[str]] = []
        groundtruth_rdfs_list: list[list[str]] = []
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
                        allow_objs_with_same_label=self.allow_objs_with_same_label,
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
                batched_obs: list[str] = []
                batched_prev_actions: list[str] = []
                batched_timestamps: list[int] = []
                graph_list: list[Data] = []
                for game_id, (step_data, graph) in game_id_to_step_data_graph.items():
                    batched_obs.append(step_data["observation"])
                    batched_prev_actions.append(step_data["previous_action"])
                    batched_timestamps.append(step_data["timestamp"])
                    graph_list.append(graph)

                # greedy decode
                results_list = self.greedy_decode(
                    collate_step_inputs(
                        self.preprocessor,
                        batched_obs,
                        batched_prev_actions,
                        batched_timestamps,
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
    ) -> list[tuple[str, ...]]:
        self.validation_step_outputs.append(self.eval_step(batch, "val"))

    def test_step(  # type: ignore
        self, batch: TWCmdGenGraphEventBatch, batch_idx: int
    ) -> list[tuple[str, ...]]:
        self.test_step_outputs.append(self.eval_step(batch, "test"))

    def wandb_log_gen_obs(
        self, outputs: list[list[tuple[str, ...]]], table_title: str
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

    def on_validation_epoch_end(self) -> None:
        generated_rdfs_list, groundtruth_rdfs_list = self.eval_free_run(
            self.trainer.datamodule.valid_free_run,  # type: ignore
            self.trainer.datamodule.val_free_run_dataloader(),  # type: ignore
        )
        self.val_free_run_f1.update(generated_rdfs_list, groundtruth_rdfs_list)
        self.log("val_free_run_f1", self.val_free_run_f1, prog_bar=True)
        self.val_free_run_em.update(generated_rdfs_list, groundtruth_rdfs_list)
        self.log("val_free_run_em", self.val_free_run_em)
        if isinstance(self.logger, WandbLogger):
            self.wandb_log_gen_obs(
                self.validation_step_outputs, "val_gen_graph_triples"
            )
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self) -> None:
        generated_rdfs_list, groundtruth_rdfs_list = self.eval_free_run(
            self.trainer.datamodule.test_free_run,  # type: ignore
            self.trainer.datamodule.test_free_run_dataloader(),  # type: ignore
        )
        self.test_free_run_f1.update(generated_rdfs_list, groundtruth_rdfs_list)
        self.log("test_free_run_f1", self.test_free_run_f1)
        self.test_free_run_em.update(generated_rdfs_list, groundtruth_rdfs_list)
        self.log("test_free_run_em", self.test_free_run_em)
        if isinstance(self.logger, WandbLogger):
            self.wandb_log_gen_obs(self.test_step_outputs, "test_gen_graph_triples")
        self.test_step_outputs.clear()

    def configure_optimizers(self) -> Optimizer:
        return AdamW(self.parameters(), lr=self.learning_rate)

    @staticmethod
    def generate_predict_table_rows(
        ids: Sequence[tuple[str, int, int]], *args: Sequence[Sequence[str]]
    ) -> list[tuple[str, ...]]:
        """Generate rows for the prediction table.

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
        """Calculate the total loss using the weighting strategy from Kendall,
        et al. 2018. with a small modification from Liebel, et al. 2018.

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
