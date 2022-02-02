import hydra
import pytorch_lightning as pl

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate


@hydra.main(config_path="train_static_label_dgu_conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(
        "Finding the optimal learning rate with the following config:\n"
        f"{OmegaConf.to_yaml(cfg)}"
    )

    # seed
    pl.seed_everything(cfg.seed)

    # trainer
    trainer = instantiate(cfg.trainer, auto_lr_find=True)

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

    # fit
    lr_finder = trainer.tuner.lr_find(lm, datamodule=dm, min_lr=1e-4)
    # plot
    fig = lr_finder.plot(suggest=True)
    fig.savefig("lr-finder.png")
    fig.show()

    # Get suggestion
    print(f"Suggestion: {lr_finder.suggestion()}")


if __name__ == "__main__":
    main()
