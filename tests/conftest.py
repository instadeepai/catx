from typing import Type

import jax
import numpy as np
import pytest
import jax.numpy as jnp
from chex import ArrayNumpy, PRNGKey

from catx.network_module import CustomHaikuNetwork
import haiku as hk

from catx.tree import TreeParameters
from catx.type_defs import (
    JaxObservations,
    Observations,
    Actions,
    Probabilities,
    Logits,
    NetworkExtras,
)


class CustomNetworkWithDropoutExtras(CustomHaikuNetwork):
    def __init__(self, depth: int) -> None:
        super().__init__(depth)
        self.network = hk.nets.MLP(
            [3] + [2 ** (self.depth + 1)], name=f"mlp_depth_{self.depth}"
        )

    def __call__(
        self,
        obs: Observations,
        key: PRNGKey,
        network_extras: NetworkExtras,
    ) -> Logits:
        return self.network(obs, dropout_rate=network_extras["dropout_rate"], rng=key)


class CustomNetworkWithoutExtras(CustomHaikuNetwork):
    def __init__(self, depth: int) -> None:
        super().__init__(depth)
        self.network = hk.nets.MLP(
            [3] + [2 ** (self.depth + 1)], name=f"mlp_depth_{self.depth}"
        )

    def __call__(
        self,
        obs: Observations,
        key: PRNGKey,
        network_extras: NetworkExtras,
    ) -> Logits:
        return self.network(obs)


# class MLPBuilder(NetworkBuilder):
#     def create_network(self, depth: int) -> hk.Module:
#         return hk.nets.MLP([3] + [2 ** (depth + 1)], name=f"mlp_depth_{depth}")


@pytest.fixture
def custom_network_without_extras() -> Type[CustomHaikuNetwork]:
    return CustomNetworkWithoutExtras


@pytest.fixture
def custom_network_with_dropout_extras() -> Type[CustomHaikuNetwork]:
    return CustomNetworkWithDropoutExtras


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
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key, num=2)
    return key
