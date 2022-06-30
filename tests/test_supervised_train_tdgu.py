import shutil
import pytest

from tdgu.supervised_train_tdgu import main
from hydra import initialize, compose


@pytest.mark.slow
def test_supervised_main(tmp_path):
    shutil.copy2("tests/data/test-fasttext.vec", tmp_path)

    with initialize(config_path="../tdgu/supervised_train_tdgu_conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                "data_module.train_path=tests/data/test_data.json",
                "data_module.train_batch_size=2",
                "data_module.val_path=tests/data/test_data.json",
                "data_module.val_batch_size=2",
                "data_module.test_path=tests/data/test_data.json",
                "data_module.test_batch_size=2",
                "model.hidden_dim=8",
                "model.text_encoder_hparams.num_conv_layers=2",
                "model.text_encoder_hparams.kernel_size=3",
                "model.graph_event_decoder_event_type_emb_dim=8",
                "model.graph_event_decoder_hidden_dim=8",
                "model.graph_event_decoder_autoregressive_emb_dim=8",
                "model.graph_event_decoder_key_query_dim=8",
                "model.text_encoder_conf.pretrained_word_embedding_path="
                f"{tmp_path}/test-fasttext.vec",
                f"+trainer.default_root_dir={tmp_path}",
                "+trainer.max_epochs=2",
            ],
        )
        main(cfg)


@pytest.mark.slow
def test_supervised_main_resume_from_ckpt(tmp_path):
    shutil.copy2("tests/data/test-fasttext.vec", tmp_path)

    with initialize(config_path="../tdgu/supervised_train_tdgu_conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                "data_module.train_path=tests/data/test_data.json",
                "data_module.train_batch_size=2",
                "data_module.val_path=tests/data/test_data.json",
                "data_module.val_batch_size=2",
                "data_module.test_path=tests/data/test_data.json",
                "data_module.test_batch_size=2",
                "model.hidden_dim=8",
                "model.text_encoder_hparams.num_conv_layers=2",
                "model.text_encoder_hparams.kernel_size=3",
                "model.graph_event_decoder_event_type_emb_dim=8",
                "model.graph_event_decoder_hidden_dim=8",
                "model.graph_event_decoder_autoregressive_emb_dim=8",
                "model.graph_event_decoder_key_query_dim=8",
                "model.text_encoder_conf.pretrained_word_embedding_path="
                f"{tmp_path}/test-fasttext.vec",
                f"+trainer.default_root_dir={tmp_path}",
                "+trainer.max_epochs=4",
                "+ckpt_path=tests/data/supervised-test.ckpt",
            ],
        )
        main(cfg)


@pytest.mark.slow
def test_supervised_main_test_only(tmp_path):
    with initialize(config_path="../tdgu/supervised_train_tdgu_conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                "data_module.test_path=tests/data/test_data.json",
                "data_module.test_batch_size=2",
                "model.hidden_dim=8",
                "model.text_encoder_hparams.num_conv_layers=2",
                "model.text_encoder_hparams.kernel_size=3",
                "model.graph_event_decoder_event_type_emb_dim=8",
                "model.graph_event_decoder_hidden_dim=8",
                "model.graph_event_decoder_autoregressive_emb_dim=8",
                "model.graph_event_decoder_key_query_dim=8",
                f"+trainer.default_root_dir={tmp_path}",
                "test_only=true",
                "+ckpt_path=tests/data/supervised-test.ckpt",
            ],
        )
        main(cfg)
