import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb

from torch.optim import AdamW, Optimizer
from typing import Dict, Tuple, Any, List, Optional, Union
from torch_geometric.data import Data, Batch
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.loggers import WandbLogger

from tdgu.train.common import TDGULightningModule
from tdgu.nn.utils import shift_tokens_right, index_edge_attr
from tdgu.nn.text import TextDecoder
from tdgu.data import TWCmdGenGraphEventStepInput, TWCmdGenObsGenBatch
from tdgu.metrics import F1
from tdgu.constants import EVENT_TYPES
from tdgu.graph import batch_to_data_list


class ObsGenSelfSupervisedTDGU(pl.LightningModule):
    """
    A LightningModule for self supervised training of the temporal discrete graph
    updater using the observation generation objective.
    """

    def __init__(
        self,
        truncated_bptt_steps: int = 1,
        text_decoder_num_blocks: int = 1,
        text_decoder_num_heads: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=["word_vocab_path", "pretrained_word_embedding_path"]
        )
        self.truncated_bptt_steps = truncated_bptt_steps

        self.tdgu = TDGULightningModule(**kwargs)
        self.text_decoder = TextDecoder(
            self.tdgu.text_encoder.get_input_embeddings(),
            text_decoder_num_blocks,
            text_decoder_num_heads,
            self.tdgu.hparams.hidden_dim,  # type: ignore
        )

        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=self.tdgu.preprocessor.pad_token_id, reduction="none"
        )

        self.val_f1 = F1()
        self.test_f1 = F1()

    def forward(  # type: ignore
        self,
        step_input: TWCmdGenGraphEventStepInput,
        step_mask: torch.Tensor,
        prev_batched_graph: Optional[Batch] = None,
    ) -> Dict[str, Union[torch.Tensor, Batch, List[torch.Tensor]]]:
        results_list = self.tdgu.greedy_decode(
            step_input,
            prev_batched_graph
            if prev_batched_graph is not None
            else Batch(
                batch=torch.empty(0, dtype=torch.long),
                x=torch.empty(0, 0, self.tdgu.preprocessor.vocab_size),
                node_label_mask=torch.empty(0, 0, dtype=torch.bool),
                node_last_update=torch.empty(0, 2, dtype=torch.long),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=torch.empty(0, 0, self.tdgu.preprocessor.vocab_size),
                edge_label_mask=torch.empty(0, 0, dtype=torch.bool),
                edge_last_update=torch.empty(0, 2, dtype=torch.long),
            ).to(self.device),
            max_event_decode_len=self.tdgu.hparams.max_event_decode_len,  # type: ignore
            max_label_decode_len=self.tdgu.hparams.max_label_decode_len,  # type: ignore
            one_hot=True,
            gumbel_tau=self.gumbel_tau,
        )

        # calculate losses with the observation
        text_decoder_output = self.text_decoder(
            shift_tokens_right(
                step_input.obs_word_ids, self.tdgu.preprocessor.bos_token_id
            ),
            step_input.obs_mask,
            results_list[-1]["batch_node_embeddings"],
            results_list[-1]["batch_node_mask"],
            step_input.prev_action_word_ids,
            step_input.prev_action_mask,
        )
        loss = torch.flatten(
            self.ce_loss(
                text_decoder_output.flatten(end_dim=1),
                step_input.obs_word_ids.flatten(),
            ).view_as(step_input.obs_word_ids)
            * step_mask.unsqueeze(-1)
        )
        # (batch * obs_len)
        return {
            "text_decoder_output": text_decoder_output,
            "loss": loss,
            "updated_batched_graph_list": [
                results["updated_batched_graph"] for results in results_list
            ],
            "decoded_event_type_id_list": [
                results["decoded_event_type_ids"] for results in results_list
            ],
            "decoded_event_src_id_list": [
                results["decoded_event_src_ids"] for results in results_list
            ],
            "decoded_event_dst_id_list": [
                results["decoded_event_dst_ids"] for results in results_list
            ],
            "decoded_event_label_word_id_list": [
                results["decoded_event_label_word_ids"] for results in results_list
            ],
            "decoded_event_label_mask_list": [
                results["decoded_event_label_mask"] for results in results_list
            ],
        }

    def on_train_epoch_start(self) -> None:
        self.game_id_to_step_data_graph: Dict[int, Tuple[Dict[str, Any], Data]] = {}

    def on_validation_epoch_start(self) -> None:
        self.game_id_to_step_data_graph = {}

    def on_test_epoch_start(self) -> None:
        self.game_id_to_step_data_graph = {}

    def training_step(  # type: ignore
        self,
        batch: TWCmdGenObsGenBatch,
        batch_idx: int,
        prev_batched_graph: Optional[Batch],
    ) -> Dict[str, Union[torch.Tensor, Batch]]:
        losses: List[torch.Tensor] = []
        for step_input, step_mask in zip(batch.step_inputs, batch.step_mask):
            # step_mask: (batch)
            results = self(step_input, step_mask, prev_batched_graph=prev_batched_graph)
            prev_batched_graph = results["updated_batched_graph_list"][-1]
            losses.append(results["loss"])
        loss = torch.stack(losses).mean()
        self.log("train_loss", loss)
        assert prev_batched_graph is not None
        return {"loss": loss, "hiddens": prev_batched_graph.detach()}

    def eval_step(self, batch: TWCmdGenObsGenBatch) -> List[Tuple[str, ...]]:
        if self.trainer.state.stage in {  # type: ignore
            RunningStage.SANITY_CHECKING,
            RunningStage.VALIDATING,
        }:
            log_prefix = "val"
        elif self.trainer.state.stage == RunningStage.TESTING:  # type: ignore
            log_prefix = "test"
        else:
            raise ValueError(
                f"Unsupported stage: {self.trainer.state.stage}"  # type: ignore
            )
        losses: List[torch.Tensor] = []
        f1 = getattr(self, f"{log_prefix}_f1")
        prev_batched_graph: Optional[Batch] = None
        # [(id, observation, previous action, decoded graph events)]
        # id = (game|walkthrough_step|random_step)
        table_data: List[Tuple[str, ...]] = []
        for ids, step_input, step_mask in zip(
            batch.ids, batch.step_inputs, batch.step_mask
        ):
            results = self(step_input, step_mask, prev_batched_graph=prev_batched_graph)
            prev_batched_graph = results["updated_batched_graph_list"][-1]
            losses.append(results["loss"])
            sizes = step_input.obs_mask.sum(dim=1).tolist()
            f1(
                [
                    word_ids[:size]
                    for word_ids, size in zip(
                        results["text_decoder_output"].argmax(dim=-1).tolist(), sizes
                    )
                ],
                [
                    word_ids[:size]
                    for word_ids, size in zip(step_input.obs_word_ids.tolist(), sizes)
                ],
            )
            table_data.extend(
                list(
                    zip(
                        "|".join(map(str, ids)),
                        self.tdgu.preprocessor.batch_decode(
                            step_input.obs_word_ids,
                            step_input.obs_mask,
                            skip_special_tokens=True,
                        ),
                        self.tdgu.preprocessor.batch_decode(
                            step_input.prev_action_word_ids,
                            step_input.prev_action_mask,
                            skip_special_tokens=True,
                        ),
                        self.decode_graph_events(
                            results["decoded_event_type_id_list"],
                            results["decoded_event_src_id_list"],
                            results["decoded_event_dst_id_list"],
                            results["decoded_event_label_word_id_list"],
                            results["decoded_event_label_mask_list"],
                            results["updated_batched_graph_list"],
                            step_mask,
                        ),
                    )
                )
            )
        loss = torch.cat(losses).mean()
        batch_size = batch.step_mask.size(1)
        self.log(f"{log_prefix}_loss", loss, batch_size=batch_size)
        self.log(f"{log_prefix}_f1", f1, batch_size=batch_size)

        return table_data

    def validation_step(  # type: ignore
        self, batch: TWCmdGenObsGenBatch, batch_idx: int
    ) -> List[Tuple[str, ...]]:
        return self.eval_step(batch)

    def test_step(  # type: ignore
        self, batch: TWCmdGenObsGenBatch, batch_idx: int
    ) -> List[Tuple[str, ...]]:
        return self.eval_step(batch)

    def tbptt_split_batch(
        self, batch: TWCmdGenObsGenBatch, split_size: int
    ) -> List[TWCmdGenObsGenBatch]:
        splits: List[TWCmdGenObsGenBatch] = []
        for i in range(0, len(batch.step_inputs), split_size):
            splits.append(
                TWCmdGenObsGenBatch(
                    ids=batch.ids[i : i + split_size],
                    step_inputs=batch.step_inputs[i : i + split_size],
                    step_mask=batch.step_mask[i : i + split_size],
                )
            )
        return splits

    @property
    def gumbel_tau(self) -> float:
        # TODO: schedule it
        return 0.5

    def configure_optimizers(self) -> Optimizer:
        return AdamW(self.parameters(), lr=self.hparams.learning_rate)  # type: ignore

    def decode_graph_events(
        self,
        event_type_id_list: List[torch.Tensor],
        src_id_list: List[torch.Tensor],
        dst_id_list: List[torch.Tensor],
        label_word_id_list: List[torch.Tensor],
        label_mask_list: List[torch.Tensor],
        batched_graph_list: List[Batch],
        step_mask: torch.Tensor,
    ) -> List[str]:
        batch_size = event_type_id_list[0].size(0)
        batched_graph_events: List[List[str]] = [[] for _ in range(batch_size)]
        for (
            event_type_ids,
            src_ids,
            dst_ids,
            batched_label_word_ids,
            batched_label_mask,
            batched_graph,
        ) in zip(
            event_type_id_list,
            src_id_list,
            dst_id_list,
            label_word_id_list,
            label_mask_list,
            batched_graph_list,
        ):
            batched_labels = self.tdgu.preprocessor.batch_decode(
                batched_label_word_ids.argmax(dim=-1),
                batched_label_mask,
                skip_special_tokens=True,
            )
            for batch_id, (
                event_type_id,
                src_id,
                dst_id,
                label,
                graph,
                is_valid_step,
            ) in enumerate(
                zip(
                    event_type_ids.tolist(),
                    src_ids,
                    dst_ids,
                    batched_labels,
                    batch_to_data_list(batched_graph, batch_size),
                    step_mask.tolist(),
                )
            ):
                if not is_valid_step:
                    continue
                node_labels = self.tdgu.preprocessor.batch_decode(
                    graph.x.argmax(dim=-1),
                    graph.node_label_mask,
                    skip_special_tokens=True,
                )
                event_type = EVENT_TYPES[event_type_id]
                src_label = "<none>"
                dst_label = "<none>"
                edge_label = "<none>"
                if event_type == "node-add":
                    src_label = label
                elif event_type == "node-delete":
                    src_label = node_labels[src_id]
                elif event_type == "edge-add":
                    src_label = node_labels[src_id]
                    dst_label = node_labels[dst_id]
                    edge_label = label
                elif event_type == "edge-delete":
                    src_label = node_labels[src_id]
                    dst_label = node_labels[dst_id]
                    edge_label_word_ids = index_edge_attr(
                        graph.edge_index,
                        graph.edge_attr.argmax(dim=-1),
                        torch.stack([src_id.unsqueeze(0), dst_id.unsqueeze(0)]),
                    )
                    edge_label_mask = index_edge_attr(
                        graph.edge_index,
                        graph.edge_label_mask,
                        torch.stack([src_id.unsqueeze(0), dst_id.unsqueeze(0)]),
                    )
                    edge_label = self.tdgu.preprocessor.batch_decode(
                        edge_label_word_ids, edge_label_mask, skip_special_tokens=True
                    )[0]
                batched_graph_events[batch_id].append(
                    f"({event_type}, {src_label}, {dst_label}, {edge_label})"
                )
        return [", ".join(graph_events) for graph_events in batched_graph_events]

    def wandb_log_gen_obs(
        self, outputs: List[List[Tuple[str, str, str, str]]], table_title: str
    ) -> None:
        eval_table_artifact = wandb.Artifact(
            table_title + f"_{self.logger.experiment.id}", "predictions"  # type: ignore
        )
        eval_table = wandb.Table(
            columns=["id", "observation", "previous action", "decoded graph events"],
            data=[item for sublist in outputs for item in sublist],
        )
        eval_table_artifact.add(eval_table, "predictions")
        self.logger.experiment.log_artifact(eval_table_artifact)  # type: ignore

    def validation_epoch_end(  # type: ignore
        self,
        outputs: List[List[Tuple[str, str, str, str]]],
    ) -> None:
        if isinstance(self.logger, WandbLogger):
            self.wandb_log_gen_obs(outputs, "val_graph_events")

    def test_epoch_end(  # type: ignore
        self,
        outputs: List[List[Tuple[str, str, str, str]]],
    ) -> None:
        if isinstance(self.logger, WandbLogger):
            self.wandb_log_gen_obs(outputs, "test_graph_events")
