import hydra
import pytorch_lightning as pl

from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")

    # seed
    pl.seed_everything(cfg.sample.data)


if __name__ == "__main__":
    main()
