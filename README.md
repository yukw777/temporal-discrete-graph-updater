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

# TDGU supports python 3.6-3.8.
# Install CUDA dependencies
python install_cuda_deps.py
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
We use [`pytest`](https://docs.pytest.org/) to write and run tests. Once your local development environment is set up, simply run `pytest` to run tests. By default, slow tests are not run locally. If you want to run them locally, use `pytest -m ""`.

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

### Git Branching Strategy
We follow a very simple git branching strategy where changes to the code base are made in a *short-lived* branch off the `main` branch. These branches should be merged as soon as possible via pull requests. A typical workflow is described below:

1. Create a new branch for your changes.
```bash
# No rules around branch names, but try to use a descriptive one.
git checkout -b <your-branch-name>
```
2. Make your changes while saving as frequently as necessary by making small commits.\
While you should always try to write descriptive commit messages, at this step, it's not strictly necessary. So commit messages like `wip` or `typo` are OK here.
2. Clean up your commits using git interactive rebasing.\
The goal of this step is to ensure all of the commits on your branch are self-contained with descriptive commit messages. You can do so by using git interactive rebasing. Here's a quick [blog post](https://www.sitepoint.com/git-interactive-rebase-guide/) and a short [video](https://www.youtube.com/watch?v=tukOm3Afd8s) on how to use it. If you'd like a more thorough documentation, you can check out this [page](https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History). Here is a good [blog post](https://cbea.ms/git-commit/) on how to write good commit messages.
2. Open a pull request (PR).\
Now that your branch is all cleaned up, go ahead and open an PR. If your branch has only one self-contained commit, you don't have to do much since the title and description would be pre-filled. If your branch has multiple self-contained commits, make sure to summarize them in the MR similar to how you'd write a git commit message.
2. Fix the PR based on review comments.\
Make sure to clean up your commits via git interactive rebasing. Your local branch may go out of sync with the remote branch at this step, and it's OK to force push `git push origin HEAD --force` to push the cleaned up branch with the fixes.
2. Merge the merge request.\
Once the merge request is approved, go ahead and merge the merge request. Squashing is recommended (unless you know what you're doing).
