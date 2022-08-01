from abc import abstractmethod

import haiku as hk

from catx.type_defs import Logits, NetworkExtras, Observations


class CATXHaikuNetwork(hk.Module):
    """An interface for implementing a custom Haiku neural network."""

    @abstractmethod
    def __init__(self, depth: int):
        """Specify the network architecture.

        The dimension of the network output layer must be; 2**(depth+1).

        Args:
           depth: the depth at which the network will be used in the tree.
        """
        super().__init__()
        self.depth = depth

    @abstractmethod
    def __call__(
        self,
        obs: Observations,
        network_extras: NetworkExtras,
    ) -> Logits:
        """Query the neural network.

        Args:
            obs: the observations, i.e., batched contexts.
            network_extras: additional information for querying the neural networks.

        Returns:
            logits: a dictionary of the networks' logits grouped pairwise.
        """
        pass
