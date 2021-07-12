import shutil

from dgu.train_static_label_dgu import main
from hydra import initialize, compose


def test_main(tmp_path):
    shutil.copy2("tests/data/test_data.json", tmp_path)
    shutil.copy2("tests/data/test-fasttext.vec", tmp_path)

    with initialize(config_path="../dgu/train_static_label_dgu_conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"data.train_path={tmp_path}/test_data.json",
                "data.train_batch_size=2",
                f"data.val_path={tmp_path}/test_data.json",
                "data.val_batch_size=2",
                f"data.test_path={tmp_path}/test_data.json",
                "data.test_batch_size=2",
                "model.hidden_dim=8",
                "model.max_num_nodes=80",
                "model.max_num_edges=100",
                "model.text_encoder_num_conv_layers=2",
                "model.text_encoder_kernel_size=3",
                "model.graph_event_decoder_key_query_dim=8",
                f"model.pretrained_word_embedding_path={tmp_path}/test-fasttext.vec",
                f"+trainer.default_root_dir={tmp_path}",
                "+trainer.max_epochs=2",
            ],
        )
        main(cfg)
