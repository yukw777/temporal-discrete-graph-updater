import hydra
import pytorch_lightning as pl

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, to_absolute_path
from pytorch_lightning.callbacks import ModelCheckpoint


@hydra.main(config_path="train_static_label_dgu_conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")
    if cfg.use_file_system_sharing_strategy:
        # on certain machines, you may run out of file descriptors, and
        # you can't increase the limit (ulimit -n). In that case you need to
        # change the sharing strategy to file_system.
        import torch.multiprocessing

        torch.multiprocessing.set_sharing_strategy("file_system")

    # seed
    pl.seed_everything(cfg.seed)

    # trainer
    trainer = instantiate(
        cfg.trainer,
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss", filename="static-label-dgu-{epoch}-{val_loss:.2f}"
            )
        ],
    )

    # data module
    dm = instantiate(cfg.data_module)

    # lightning module
    lm = instantiate(
        cfg.model,
        **cfg.train,
        word_vocab_path=cfg.data_module.word_vocab_path,
        node_vocab_path=cfg.data_module.node_vocab_path,
        relation_vocab_path=cfg.data_module.relation_vocab_path,
    )

    if cfg.test_only:
        trainer.test(
            model=lm,
            ckpt_path=to_absolute_path(cfg.trainer.resume_from_checkpoint),
            datamodule=dm,
        )
        return

    # fit
    trainer.fit(lm, datamodule=dm)

    # test
    trainer.test()


if __name__ == "__main__":
    main()
