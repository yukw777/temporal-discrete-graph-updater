import torch
import torch.nn as nn
import pytorch_lightning as pl

from torch.optim import AdamW, Optimizer
from typing import Dict, Tuple, Any, List, Optional
from torch_geometric.data import Data, Batch
from hydra.utils import to_absolute_path
from pathlib import Path
from pytorch_lightning.trainer.states import RunningStage

from tdgu.train.common import TDGULightningModule
from tdgu.nn.utils import load_fasttext, shift_tokens_right
from tdgu.graph import data_list_to_batch, batch_to_data_list
from tdgu.nn.text import TextDecoder
from tdgu.preprocessor import SpacyPreprocessor, PAD, UNK, BOS, EOS
from tdgu.data import TWCmdGenGraphEventStepInput, TWCmdGenGraphEventDataCollator
from tdgu.metrics import F1


class ObsGenSelfSupervisedTDGU(pl.LightningModule):
    """
    A LightningModule for self supervised training of the temporal discrete graph
    updater using the observation generation objective.
    """

    def __init__(
        self,
        word_vocab_path: Optional[str] = None,
        pretrained_word_embedding_path: Optional[str] = None,
        text_decoder_num_blocks: int = 1,
        text_decoder_num_heads: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=["word_vocab_path", "pretrained_word_embedding_path"]
        )

        # preprocessor
        self.preprocessor = (
            SpacyPreprocessor([PAD, UNK, BOS, EOS])
            if word_vocab_path is None
            else SpacyPreprocessor.load_from_file(to_absolute_path(word_vocab_path))
        )

        # pretrained word embeddings
        pretrained_word_embeddings: Optional[nn.Embedding]
        if pretrained_word_embedding_path is None:
            pretrained_word_embeddings = None
        else:
            abs_pretrained_word_embedding_path = Path(
                to_absolute_path(pretrained_word_embedding_path)
            )
            serialized_path = abs_pretrained_word_embedding_path.parent / (
                abs_pretrained_word_embedding_path.stem + ".pt"
            )
            pretrained_word_embeddings = load_fasttext(
                str(abs_pretrained_word_embedding_path),
                serialized_path,
                self.preprocessor.get_vocab(),
                self.preprocessor.pad_token_id,
            )
            pretrained_word_embeddings.requires_grad_(requires_grad=False)

        self.tdgu = TDGULightningModule(
            text_encoder_vocab_size=self.preprocessor.vocab_size,
            label_head_bos_token_id=self.preprocessor.bos_token_id,
            label_head_eos_token_id=self.preprocessor.eos_token_id,
            label_head_pad_token_id=self.preprocessor.pad_token_id,
            pretrained_word_embeddings=pretrained_word_embeddings,
            **kwargs,
        )
        self.text_decoder = TextDecoder(
            self.tdgu.text_encoder.get_input_embeddings(),
            text_decoder_num_blocks,
            text_decoder_num_heads,
            self.tdgu.hparams.hidden_dim,  # type: ignore
        )

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.preprocessor.pad_token_id)

        self.val_f1 = F1()
        self.test_f1 = F1()

    def forward(  # type: ignore
        self, step_input: TWCmdGenGraphEventStepInput, batched_graph: Batch
    ) -> torch.Tensor:
        results_list = self.tdgu.greedy_decode(
            step_input,
            batched_graph,
            max_event_decode_len=self.tdgu.hparams.max_event_decode_len,  # type: ignore
            max_label_decode_len=self.tdgu.hparams.max_label_decode_len,  # type: ignore
            one_hot=True,
            gumbel_tau=self.gumbel_tau,
        )

        # update graphs in game_id_to_step_data_graph
        for (game_id, (step_data, _)), updated_graph in zip(
            self.game_id_to_step_data_graph.items(),
            batch_to_data_list(
                results_list[-1]["updated_batched_graph"],
                step_input.obs_word_ids.size(0),
            ),
        ):
            self.game_id_to_step_data_graph[game_id] = (
                step_data,
                updated_graph.detach(),
            )

        # calculate losses with the observation
        return self.text_decoder(
            shift_tokens_right(step_input.obs_word_ids, self.preprocessor.bos_token_id),
            step_input.obs_mask,
            results_list[-1]["batch_node_embeddings"],
            results_list[-1]["batch_node_mask"],
            step_input.prev_action_word_ids,
            step_input.prev_action_mask,
        )

    def on_train_epoch_start(self) -> None:
        self.game_id_to_step_data_graph: Dict[int, Tuple[Dict[str, Any], Data]] = {}

    def on_validation_epoch_start(self) -> None:
        self.game_id_to_step_data_graph = {}

    def on_test_epoch_start(self) -> None:
        self.game_id_to_step_data_graph = {}

    def prepare_greedy_decode_input(
        self,
        batch: List[Tuple[int, Dict[str, Any]]],
        collator: TWCmdGenGraphEventDataCollator,
    ) -> Tuple[TWCmdGenGraphEventStepInput, Batch]:
        # finished games are the ones that were in game_id_to_graph, but are not
        # part of the new batch
        for finished_game_id in self.game_id_to_step_data_graph.keys() - {
            game_id for game_id, _ in batch
        }:
            self.game_id_to_step_data_graph.pop(finished_game_id)

        # new games are the ones that were not in game_id_to_graph, but are now
        # part of the new batch.
        # due to Python's dictionary ordering (insertion order), new games are
        # added always to the end.
        for game_id, step_data in batch:
            if game_id in self.game_id_to_step_data_graph:
                _, graph = self.game_id_to_step_data_graph[game_id]
                self.game_id_to_step_data_graph[game_id] = (step_data, graph)
            else:
                self.game_id_to_step_data_graph[game_id] = (
                    step_data,
                    Data(
                        x=torch.empty(0, 0, self.preprocessor.vocab_size),
                        node_label_mask=torch.empty(0, 0).bool(),
                        node_last_update=torch.empty(0, 2, dtype=torch.long),
                        edge_index=torch.empty(2, 0, dtype=torch.long),
                        edge_attr=torch.empty(0, 0, self.preprocessor.vocab_size),
                        edge_label_mask=torch.empty(0, 0).bool(),
                        edge_last_update=torch.empty(0, 2, dtype=torch.long),
                    ).to(self.device),
                )

        # construct a batch
        batched_obs: List[str] = []
        batched_prev_actions: List[str] = []
        batched_timestamps: List[int] = []
        graph_list: List[Data] = []
        for game_id, (step_data, graph) in self.game_id_to_step_data_graph.items():
            batched_obs.append(step_data["observation"])
            batched_prev_actions.append(step_data["previous_action"])
            batched_timestamps.append(step_data["timestamp"])
            graph_list.append(graph)
        return collator.collate_step_inputs(
            batched_obs, batched_prev_actions, batched_timestamps
        ).to(self.device), data_list_to_batch(graph_list)

    def training_step(  # type: ignore
        self, batch: List[Tuple[int, Dict[str, Any]]], batch_idx: int
    ) -> torch.Tensor:
        step_input, batched_graph = self.prepare_greedy_decode_input(
            batch, self.trainer.datamodule.collator  # type: ignore
        )
        text_decoder_output = self(step_input, batched_graph)
        # (batch, obs_len, num_word)
        return self.ce_loss(
            text_decoder_output.flatten(end_dim=1), step_input.obs_word_ids.flatten()
        )

    def eval_step(self, batch: List[Tuple[int, Dict[str, Any]]]) -> None:
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
        step_input, batched_graph = self.prepare_greedy_decode_input(
            batch, self.trainer.datamodule.collator  # type: ignore
        )
        batch_size = step_input.obs_word_ids.size(0)
        text_decoder_output = self(step_input, batched_graph)
        loss = self.ce_loss(
            text_decoder_output.flatten(end_dim=1), step_input.obs_word_ids.flatten()
        )
        self.log(f"{log_prefix}_loss", loss, batch_size=batch_size)
        sizes = step_input.obs_mask.sum(dim=1).tolist()
        f1 = getattr(self, f"{log_prefix}_f1")
        f1(
            [
                ids[:size]
                for ids, size in zip(text_decoder_output.argmax(dim=-1).tolist(), sizes)
            ],
            [ids[:size] for ids, size in zip(step_input.obs_word_ids.tolist(), sizes)],
        )
        self.log(f"{log_prefix}_f1", f1, batch_size=batch_size)

    def validation_step(  # type: ignore
        self, batch: List[Tuple[int, Dict[str, Any]]], batch_idx: int
    ) -> None:
        self.eval_step(batch)

    def test_step(  # type: ignore
        self, batch: List[Tuple[int, Dict[str, Any]]], batch_idx: int
    ) -> None:
        self.eval_step(batch)

    @property
    def gumbel_tau(self) -> float:
        # TODO: schedule it
        return 0.5

    def configure_optimizers(self) -> Optimizer:
        return AdamW(self.parameters(), lr=self.hparams.learning_rate)  # type: ignore
