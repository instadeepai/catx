from abc import abstractmethod
import haiku as hk
from chex import PRNGKey
from catx.type_defs import Observations, StateExtras, Logits


class CustomHaikuNetwork(hk.Module):
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
        pass

    @abstractmethod
    def __call__(
        self, obs: Observations, state_extras: StateExtras, key: PRNGKey
    ) -> Logits:
        """Query the neural network.

        Args:
            obs: the observations, i.e., batched contexts.
            state_extras: additional information for querying the neural networks.
            key: pseudo-random number generator.

        Returns:
            logits: a dictionary of the networks' logits grouped pairwise.
        """
        pass
