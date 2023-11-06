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

[![Paper](http://img.shields.io/badge/paper-arxiv.2311.01928-B31B1B.svg)](https://arxiv.org/abs/2311.01928)

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
# Install poetry https://python-poetry.org/
curl -sSL https://install.python-poetry.org | python3 -

# Clone the repository
git clone https://github.com/yukw777/temporal-discrete-graph-updater
cd temporal-discrete-graph-updater

# Install TDGU using poetry
poetry install
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
python scripts/main.py fit \
--model config/supervised/main-model.yaml \
--data config/supervised/data.yaml
--trainer.accelerator gpu
```

If you want to log to [Weights & Biases](https://wandb.ai/), simply set the logger.

```bash
python scripts/main.py fit \
--model config/supervised/main-model.yaml \
--data config/supervised/data.yaml
--trainer.accelerator gpu \
--config config/supervised/wandb-logger.yaml
```

## Configuration

This project uses [PyTorch Lightning CLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html) as its configuration system. Take a look at [config](config) for the default configuration, which can be used to reproduce the main results of the paper.

## Developing

### Local Development Environment Setup

```bash
# Install poetry https://python-poetry.org/
curl -sSL https://install.python-poetry.org | python3 -

# Clone the repository
git clone https://github.com/yukw777/temporal-discrete-graph-updater
cd temporal-discrete-graph-updater

# Install TDGU with dev dependencies using poetry
poetry install --with dev
```

### Testing

We use [`pytest`](https://docs.pytest.org/) to write and run tests. Once your local development environment is set up, simply run `pytest` to run tests. By default, slow tests are not run locally. If you want to run them locally, use `pytest -m ""`.

All the tests are run as part of the CI/CD pipeline, so it's highly recommended that you make sure all the tests locally before opening a merge request.

### Code Quality

We use various tools to ensure code quality. All of these tools can be run automatically at every commit by using [pre-commit](https://pre-commit.com/). Simply run `pre-commit install` after installing TDGU to set it up. You can also set these tools up with your favorite IDE. Please see below for instructions.

#### Code Linter

We use [`flake8`](https://flake8.pycqa.org/) to lint our code. While you can manually run it, it's highly recommended that you set up your text editor or IDE to run it automatically after each save. See the list below for documentations on how to set `flake8` up for various text editors and IDEs.

- [VSCode](https://code.visualstudio.com/docs/python/linting)

#### Code Formatter

We use [`black`](https://github.com/psf/black) to automatically format our code. While you can manually run it, it's highly recommended that you set up your text editor or IDE to run it automatically after each save. See the list below for documentations on how to set `black` up for various text editors and IDEs.

- [VSCode](https://dev.to/adamlombard/how-to-use-the-black-python-code-formatter-in-vscode-3lo0)

#### Static Type Checker

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

We follow a very simple git branching strategy where changes to the code base are made in a _short-lived_ branch off the `main` branch. These branches should be merged as soon as possible via pull requests. A typical workflow is described below:

1. Create a new branch for your changes.

```bash
# No rules around branch names, but try to use a descriptive one.
git checkout -b <your-branch-name>
```

2. Make your changes while saving as frequently as necessary by making small commits.\
   While you should always try to write descriptive commit messages, at this step, it's not strictly necessary. So commit messages like `wip` or `typo` are OK here.
3. Clean up your commits using git interactive rebasing.\
   The goal of this step is to ensure all of the commits on your branch are self-contained with descriptive commit messages. You can do so by using git interactive rebasing. Here's a quick [blog post](https://www.sitepoint.com/git-interactive-rebase-guide/) and a short [video](https://www.youtube.com/watch?v=tukOm3Afd8s) on how to use it. If you'd like a more thorough documentation, you can check out this [page](https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History). Here is a good [blog post](https://cbea.ms/git-commit/) on how to write good commit messages.
4. Open a pull request (PR).\
   Now that your branch is all cleaned up, go ahead and open an PR. If your branch has only one self-contained commit, you don't have to do much since the title and description would be pre-filled. If your branch has multiple self-contained commits, make sure to summarize them in the MR similar to how you'd write a git commit message.
5. Fix the PR based on review comments.\
   Make sure to clean up your commits via git interactive rebasing. Your local branch may go out of sync with the remote branch at this step, and it's OK to force push `git push origin HEAD --force` to push the cleaned up branch with the fixes.
6. Merge the merge request.\
   Once the merge request is approved, go ahead and merge the merge request. Squashing is recommended (unless you know what you're doing).
