from abc import ABC, abstractmethod

import haiku as hk


class NetworkBuilder(ABC):
    """An interface for implementing a neural network builder."""

    @abstractmethod
    def create_network(self, depth: int) -> hk.Module:
        """An abstract method for creating a Haiku neural network.

        The dimension of the network output layer should be 2**(depth+1)

        Args:
            depth: the depth at which the network will be used in the tree.

        Returns:
            a Haiku neural network module.
        """

        pass
