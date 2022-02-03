<div align="center">

# Temporal Discrete Graph Updater

<!--
Badges upon publication
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
ARXIV
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
![CI testing](https://github.com/yukw777/pl-hydra-seed/actions/workflows/ci-testing.yml/badge.svg)


<!--
Conference
-->
</div>

## Description
The Temporal Discrete Graph Updater (TDGU) is a text-to-graph model that incrementally constructs temporal dynamic knowledge graphs from interactive text-based games.

## Quickstart
Install dependencies.
```bash
# Clone the repository
git clone https://github.com/yukw777/temporal-discrete-graph-updater
cd temporal-discrete-graph-updater

# Install dependencies
# TDGU supports python 3.6-3.8.
# For CUDA 10.2:
pip install torch-scatter torch-sparse torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu102.html
# For CUDA 11.3
pip install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install torch-scatter torch-sparse torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu113.html

# Install the tdgu module in the editable mode
pip install -e .
```

Download the dataset.
```bash
cd temporal-discrete-graph-updater
mkdir data
cd data
wget https://aka.ms/twkg/cmd_gen.0.2.zip
unzip cmd_gen.0.2.zip -d cmd_gen.0.2
```

Download the pretrained embeddings.
```bash
cd temporal-discrete-graph-updater
mkdir embedding
cd embedding
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
unzip crawl-300d-2M.vec.zip
```

Next, run the training script.
```bash
python -m tdgu.train_tdgu \
+trainer.gpus=1 # use one gpu
```

If you want to log to [Weights & Biases](https://wandb.ai/), simply set the logger.
```bash
python -m tdgu.train_tdgu \
trainer/logger=wandb \
+trainer.gpus=1
```

## Configuration
This project uses [Hydra](https://hydra.cc/) as its configuration system. Take a look at [project/conf](project/conf) for the default configuration, which can be used to reproduce the main results of the paper.

## Developing
### Local Development Environment Setup
```bash
# Clone the repository
git clone https://github.com/yukw777/temporal-discrete-graph-updater
cd temporal-discrete-graph-updater

# Create a virtualenv with python 3.8 (recommended)
# NOTE: this is just an example using virtualenv. Use your tool of choice for creating a python virtual environment.
virtualenv -p python3.8 venv
source venv/bin/activate

# Install dependencies
# TDGU supports python 3.6-3.8.
# For CUDA 10.2:
pip install torch-scatter torch-sparse torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu102.html
# For CUDA 11.3
pip install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install torch-scatter torch-sparse torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu113.html

# Install the tdgu module in the editable mode
pip install -e .

# Install development requirements
pip install -r requirements-dev.txt
```

### Testing
We use [`pytest`](https://docs.pytest.org/) to write and run tests. Once your local development environment is set up, simply run `pytest` to run tests.

All the tests are run as part of the CI/CD pipeline, so it's highly recommended that you make sure all the tests locally before opening a merge request.

### Code Linter
We use [`flake8`](https://flake8.pycqa.org/) to lint our code. While you can manually run it, it's highly recommended that you set up your text editor or IDE to run it automatically after each save. See the list below for documentations on how to set `flake8` up for various text editors and IDEs.

- [VSCode](https://code.visualstudio.com/docs/python/linting)

### Code Formatter
We use [`black`](https://github.com/psf/black) to automatically format our code. While you can manually run it, it's highly recommended that you set up your text editor or IDE to run it automatically after each save. See the list below for documentations on how to set `black` up for various text editors and IDEs.

- [VSCode](https://dev.to/adamlombard/how-to-use-the-black-python-code-formatter-in-vscode-3lo0)

### Static Type Checker
We use [`mypy`](http://mypy-lang.org/) as our static type checker. It uses Python's [type hints](https://docs.python.org/3.9/library/typing.html) to perform static type checks. Please note that we use python3.9 which has some new type hints that do not exist in the previous versions. While you can manually run it, it's highly recommended that you set up your text editor or IDE to run it automatically after each save. See the list below for documentations on how to set `mypy` up for various text editors and IDEs.

- [VSCode](https://code.visualstudio.com/docs/python/linting#_mypy)

<!-- ## Citation
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
``` -->
