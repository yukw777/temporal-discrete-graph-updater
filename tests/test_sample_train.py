from dgu.sample_train import main

from hydra.experimental import initialize, compose


def test_main():
    with initialize(config_path="../dgu/conf"):
        cfg = compose(config_name="config")
        main(cfg)
