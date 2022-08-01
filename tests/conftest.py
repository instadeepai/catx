from typing import Type

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from chex import ArrayNumpy, PRNGKey

from catx.network_module import CATXHaikuNetwork
from catx.tree import TreeParameters
from catx.type_defs import (
    Actions,
    JaxObservations,
    Logits,
    NetworkExtras,
    Observations,
    Probabilities,
)


class CatxNetworkWithDropoutExtras(CATXHaikuNetwork):
    def __init__(self, depth: int) -> None:
        super().__init__(depth)
        self.network = hk.nets.MLP(
            [3] + [2 ** (self.depth + 1)], name=f"mlp_depth_{self.depth}"
        )

    def __call__(
        self,
        obs: Observations,
        network_extras: NetworkExtras,
    ) -> Logits:
        return self.network(
            obs, dropout_rate=network_extras["dropout_rate"], rng=hk.next_rng_key()
        )


class CatxNetworkWithoutExtras(CATXHaikuNetwork):
    def __init__(self, depth: int) -> None:
        super().__init__(depth)
        self.network = hk.nets.MLP(
            [3] + [2 ** (self.depth + 1)], name=f"mlp_depth_{self.depth}"
        )

    def __call__(
        self,
        obs: Observations,
        network_extras: NetworkExtras,
    ) -> Logits:
        return self.network(obs)


@pytest.fixture
def catx_network_without_extras() -> Type[CATXHaikuNetwork]:
    return CatxNetworkWithoutExtras


@pytest.fixture
def catx_network_with_dropout_extras() -> Type[CATXHaikuNetwork]:
    return CatxNetworkWithDropoutExtras


@pytest.fixture
def tree_parameters() -> TreeParameters:
    tree_param = TreeParameters.construct(bandwidth=1 / 4, discretization_parameter=8)

    return tree_param


@pytest.fixture
def observations() -> Observations:
    obs = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])

    return obs


@pytest.fixture
def jax_observations(observations: Observations) -> JaxObservations:
    obs = jnp.asarray(observations)

    return obs


@pytest.fixture
def actions() -> Actions:
    actions = np.array([0.564849, 0.57515746, 0.6175556, 0.05339688])

    return actions


@pytest.fixture
def probabilities() -> Probabilities:
    probabilities = np.array([3.85, 3.85, 3.85, 3.85], dtype=float)

    return probabilities


@pytest.fixture
def costs() -> ArrayNumpy:
    costs = np.array([0.06672329, 0.22529681, 0.03357587, 0.16503502])

    return costs


@pytest.fixture
def epsilon() -> float:
    return 0.0


@pytest.fixture
def key() -> PRNGKey:
    return jax.random.PRNGKey(42)
