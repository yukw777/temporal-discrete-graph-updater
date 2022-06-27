import os

import hydra
import pytorch_lightning as pl

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, to_absolute_path
from pytorch_lightning.callbacks import ModelCheckpoint


@hydra.main(config_path="self_supervised_train_tdgu_conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")

    # seed
    pl.seed_everything(cfg.seed)

    # trainer
    trainer = instantiate(
        cfg.trainer,
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                filename="tdgu-self-supervised-obs-gen-{epoch}-{val_loss:.2f}",
            )
        ],
    )

    # data module
    dm = instantiate(cfg.data_module)

    # checkpoint path
    ckpt_path = to_absolute_path(cfg.ckpt_path) if "ckpt_path" in cfg else None

    if ckpt_path is not None:
        # don't load the pretrained embeddings since we're loading from a checkpoint
        del cfg.model.pretrained_word_embedding_path

    # lightning module
    lm = instantiate(
        cfg.model, **cfg.train, word_vocab_path=cfg.data_module.word_vocab_path
    )

    if cfg.test_only:
        trainer.test(model=lm, datamodule=dm, ckpt_path=ckpt_path)
        return

    # fit
    trainer.fit(lm, datamodule=dm, ckpt_path=ckpt_path)

    # test
    trainer.test(ckpt_path="best", datamodule=dm)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    main()
