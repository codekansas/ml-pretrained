About
=====

This package is designed to make it easy to organize your machine learning projects
and quickly implement a project idea. It is similar to something like
`PyTorch Lightning <https://www.pytorchlightning.ai>`_ or `Keras <https://keras.io>`_,
but it is a bit more opinionated, making it easier to get a project off the ground.

The closest similar thing would be something like `this <https://github.com/ashleve/lightning-hydra-template>`_,
but I wanted to avoid using PyTorch Lightning or Hydra directly, since they
have been a bit unweildy in my experience. I tried to keep the amount of
magic to a minimum; I use `OmegaConf <https://omegaconf.readthedocs.io>`_ for
dealing with config files, and the framework expects a specific project structure
so that it can do some caching for associating the objects referenced in the
config file with their implementations, but other than that it's just a bunch
of standard PyTorch code.

I built this almost entirely for my own purposes, to address some of the pain points
I had while using PyTorch Lightning and Fairseq. If you're reading this then I hope
you find it useful as well!
