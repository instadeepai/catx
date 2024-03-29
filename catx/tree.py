from typing import TYPE_CHECKING, Dict, Optional, Type

import haiku as hk
import jax.numpy as jnp
import numpy as np
from chex import Array

from catx.network_module import CATXHaikuNetwork
from catx.type_defs import Logits, NetworkExtras

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass


@dataclass
class TreeParameters:
    """Dataclass for holding the tree parameters.

    Args:
        bandwidth: the bucket half width covered by action centroid.
        discretization_parameter: the number of action centroids.
        action_space: the range from which actions can be generated.
        depth: number of layers in the tree.
        spaces: an array indicating the start and end range of each action centroid.
        volumes: an array indicating the bandwidth of around each action centroid.
        probabilities: h-smoothing of policy π_t (one over volumes).
    """

    bandwidth: float
    discretization_parameter: int
    action_space: Array
    depth: int
    spaces: Array
    volumes: Array
    probabilities: Array

    @classmethod
    def construct(
        cls, discretization_parameter: int, bandwidth: float
    ) -> "TreeParameters":
        """A constructor for calculating and initializing the tree parameters.

        Args:
            discretization_parameter: the number of action centroids.
            bandwidth: the bucket half width covered by action centroid.

        Returns:
            An initialized TreeParameters dataclass.
        """

        assert bandwidth > 0, "Only positive bandwidth value is admissible."
        if discretization_parameter < 2 or (
            discretization_parameter & (discretization_parameter - 1)
        ):
            raise ValueError(
                "discretization_parameter must be a power of 2 number and larger than 1."
            )

        action_space = jnp.linspace(
            bandwidth, 1 - bandwidth, num=discretization_parameter
        )
        depth = int(np.log2(discretization_parameter))
        spaces = jnp.clip(
            (
                action_space.reshape((-1, 1))
                + jnp.asarray([-bandwidth, bandwidth]).reshape((1, -1))
            ),
            a_min=0,
            a_max=1,
        )
        volumes = jnp.diff(spaces).flatten()
        probabilities = 1 / volumes

        # noinspection PyArgumentList
        return cls(
            bandwidth=bandwidth,
            discretization_parameter=discretization_parameter,
            action_space=action_space,
            depth=depth,
            spaces=spaces,
            volumes=volumes,
            probabilities=probabilities,
        )


class Tree(hk.Module):
    def __init__(
        self,
        catx_network: Type[CATXHaikuNetwork],
        tree_params: TreeParameters,
        name: Optional[str] = None,
    ):
        """The tree as a JAX Haiku module.

        Args:
            catx_network: class specifying the neural network architecture.
            tree_params: object holding the tree parameters.
            name: name of the tree.
        """

        super().__init__(name)
        self.tree_params = tree_params

        self.networks = {
            depth: catx_network(depth=depth) for depth in range(self.tree_params.depth)
        }

    def __call__(self, obs: Array, network_extras: NetworkExtras) -> Dict[int, Logits]:
        """Query the neural networks of the tree.

        Args:
            obs: the observations, i.e., batched contexts.
            network_extras: additional information for querying the neural networks.

        Returns:
            logits: a dictionary of the networks' logits grouped pairwise.
        """

        logits = {}
        for i in range(self.tree_params.depth):
            n_leafs = 2 ** (i + 1)
            c = self.networks[i](
                obs=obs,
                network_extras=network_extras,
            ).reshape(-1, n_leafs // 2, 2)
            logits[i] = c
        return logits
