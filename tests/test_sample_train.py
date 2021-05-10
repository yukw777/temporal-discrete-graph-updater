# TODO: change package name
from project.sample_train import main

from hydra.experimental import initialize, compose


def test_main():
    with initialize(config_path="../project/conf"):
        cfg = compose(config_name="config")
        main(cfg)
