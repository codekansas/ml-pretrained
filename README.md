<div align="center">

# ML Pretrained

[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
<br />
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![ruff](https://img.shields.io/badge/Linter-Ruff-red.svg?labelColor=gray)](https://github.com/charliermarsh/ruff)
<br />
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/codekansas/ml-pretrained/blob/master/LICENSE)

</div>

<br />

```bash
$ pip install ml-pretrained
# Try out the RWKV model. This requires the `tokenizers` package.
$ pip install tokenizers
$ python -m pretrained.rwkv 430m 'Scientists recently discovered an island populated entirely by cats. To their astonishment, none of the cats were hungry. Upon further investigation,'
Scientists recently discovered an island populated entirely by cats. To their astonishment, none of the cats were hungry. Upon further investigation, in fact, researchers discovered that they were members of an ancestral, living species. When this research was published in 2009 in the journal Current Biology, there was widespread praise for their research.

So what do we make of the current understanding of what happens to these amazing creatures when they no longer exist on a cold planet? Much of the literature on the matter seems to conclude that there’s no hope of coming to grips with this question, that everything we know about this species, or even any of the non-human species that may be out there in space, is simply a human’s or animal’s invention.
# Try out the Tacotron2 model. This requires the `sounddevice`, `inflect` and `ftfy` packages.
$ pip install sounddevice inflect ftfy
$ python -m pretrained.tacotron2 'Scientists recently discovered an island populated entirely by cats. To their astonishment, none of the cats were hungry.'
# Chain these together.
$ python -m pretrained.tacotron2 "$(python -m pretrained.rwkv 430m 'Scientists recently discovered an island populated entirely by cats. To their surprise,' --tsz 64)"
```

## What is this?

This is a collection of pre-trained model implementations, which can be used in down-stream packages.

### Goal

The goal of this repository is to make it as easy as possible to try out a pre-trained model, and eventually incorporate it into a new project. To that end, each implementation is a self-contained file, so that you can just copy-paste it wholesale into whatever project your using or use it directly without adding a bunch of dependencies. Additionally, the implementations use high-quality, modern Python syntax to be as easy to follow as possible.

This also includes some custom Triton kernels for certain operations which may benefit from GPU acceleration.

### License

Note that while this particular code is MIT licensed, but that does not mean that all of the model weights are. You should comply with the upstream licenses for models that you use.
