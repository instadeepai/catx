![Logo_CatX_Final_PNG](docs/img/Logo_CatX_Final_PNG.png)

CATX is a library for training and using contextual bandits in a continuous action space.

CATX builds on the work presented in
["Efficient Contextual Bandits with Continuous Actions (CATS)"](https://arxiv.org/pdf/2006.06040.pdf) by Majzoubi et al.
CATX brings forth the freedom to implement custom neural network architectures
as decision agents within the learning algorithm.
It allows for greater scalability and context modalities while
also leveraging the computational speed of [JAX](https://github.com/google/jax).

## Target users
CATX is aimed at users facing continuous action contextual bandits problems - any problem where you need to take
continuous actions while maximising the desired reward (and, consequently, minimising cost, time or effort expenditures).
Contextual bandits settings, where the exploration-exploitation trade-off needs to be dealt with,
can be found in many industries and use cases.
CATX offers a valuable boost to this type of problem, by implementing contextual bandits with continuous actions in JAX,
and allowing custom neural networks in the tree structure of the CATS algorithm.



## Documentation
Go to [documentation](https://catx.readthedocs.io/en/main/)
to find everything you need to know about CATX
from the [installation](https://catx.readthedocs.io/en/main/installation/) with `pip install catx`
to a quick [getting started](https://catx.readthedocs.io/en/main/getting_started/) example
and much more on CATX and its inner workings.


## Citing CATX

```
@software{catx2022github,
  author = {Wissam Bejjani and Cyprien Courtot},
  title = {CATX: contextual bandits library for Continuous Action Trees with smoothing in JAX},
  url = {https://github.com/instadeepai/catx/},
  version = {0.2.1},
  year = {2022},
}
```

In this bibtex entry, the version number is intended to be from
[`catx/VERSION`](https://github.com/instadeepai/catx/blob/main/catx/VERSION),
and the year corresponds to the project's open-source release.
