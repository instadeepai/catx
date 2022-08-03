![Logo_CatX_Final_PNG](img/Logo_CatX_Final_PNG.png)

# CATX: contextual bandits for continuous actions using trees with smoothing in JAX

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
