<div align="center">

# Discrete Graph Updater

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
<!--
ARXIV
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
![CI testing](https://github.com/yukw777/pl-hydra-seed/actions/workflows/ci-testing.yml/badge.svg)


<!--
Conference
-->
</div>

## Description
What it does

## How to run
First, install dependencies
```bash
# clone
$ git clone https://github.com/yukw777/discrete-graph-updater
$ cd discrete-graph-updater

# install dependencies with pre-built wheels to speed up installation
# we use CUDA 10.2, which is the default CUDA version that comes packaged with PyTorch
$ export CUDA=cu102
$ pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
$ pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
$ pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
$ pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
$ pip install -e .
```
Next, run the training module.
```bash
python -m project.sample_train
```

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:
```python
from project.sample_train import main

main()
```

## Configuration
This project uses [Hydra](https://hydra.cc/) as its configuration system. Take a look at [project/conf](project/conf).

## Citation
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```
