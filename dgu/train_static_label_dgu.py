import hydra
import pytorch_lightning as pl

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from dgu.data import TWCmdGenTemporalDataModule
from dgu.nn.graph_updater import StaticLabelDiscreteGraphUpdater


@hydra.main(config_path="train_static_label_dgu_conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")

    # seed
    pl.seed_everything(cfg.seed)

    # trainer
    trainer = instantiate(cfg.trainer)

    # data module
    dm = TWCmdGenTemporalDataModule(
        **cfg.data,
        max_num_nodes=cfg.model.max_num_nodes,
        max_num_edges=cfg.model.max_num_edges,
    )

    # lightning module
    lm = StaticLabelDiscreteGraphUpdater(
        **cfg.model,
        **cfg.train,
        word_vocab_path=cfg.data.word_vocab_file,
        node_vocab_path=cfg.data.node_vocab_file,
        relation_vocab_path=cfg.data.relation_vocab_file,
    )

    # fit
    trainer.fit(lm, datamodule=dm)


if __name__ == "__main__":
    main()
