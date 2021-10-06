import shutil
import pytest

from dgu.train_static_label_dgu import main
from hydra import initialize, compose


@pytest.mark.parametrize("use_file_system_sharing_strategy", ["true", "false"])
def test_main(tmp_path, use_file_system_sharing_strategy):
    shutil.copy2("tests/data/test-fasttext.vec", tmp_path)

    with initialize(config_path="../dgu/train_static_label_dgu_conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"use_file_system_sharing_strategy={use_file_system_sharing_strategy}",
                "data_module.train_path=tests/data/test_data.json",
                "data_module.train_batch_size=2",
                "data_module.val_path=tests/data/test_data.json",
                "data_module.val_batch_size=2",
                "data_module.test_path=tests/data/test_data.json",
                "data_module.test_batch_size=2",
                "model.hidden_dim=8",
                "model.text_encoder_num_conv_layers=2",
                "model.text_encoder_kernel_size=3",
                "model.graph_event_decoder_hidden_dim=8",
                "model.graph_event_decoder_key_query_dim=8",
                f"model.pretrained_word_embedding_path={tmp_path}/test-fasttext.vec",
                f"+trainer.default_root_dir={tmp_path}",
                "+trainer.max_epochs=2",
            ],
        )
        main(cfg)


def test_main_test_only(tmp_path):
    with initialize(config_path="../dgu/train_static_label_dgu_conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                "data_module.test_path=tests/data/test_data.json",
                "data_module.test_batch_size=2",
                "model.hidden_dim=8",
                "model.text_encoder_num_conv_layers=2",
                "model.text_encoder_kernel_size=3",
                "model.graph_event_decoder_key_query_dim=8",
                f"+trainer.default_root_dir={tmp_path}",
                "test_only=true",
                "+trainer.resume_from_checkpoint=tests/data/test.ckpt",
            ],
        )
        main(cfg)
