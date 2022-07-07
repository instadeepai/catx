import numpy as np
import pytest
import jax.numpy as jnp
from chex import ArrayNumpy

from catx.network_builder import NetworkBuilder
import haiku as hk

from catx.tree import TreeParameters
from catx.type_defs import JaxObservations, Observations, Actions, Probabilities


class MLPBuilder(NetworkBuilder):
    def create_network(self, depth: int) -> hk.Module:
        return hk.nets.MLP([3] + [2 ** (depth + 1)], name=f"mlp_depth_{depth}")


@pytest.fixture
def mlp_builder() -> MLPBuilder:
    builder = MLPBuilder()

    return builder


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
