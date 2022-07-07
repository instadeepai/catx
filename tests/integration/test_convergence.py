from typing import List, Tuple

import haiku as hk
import jax
import optax
import pytest
from jax import numpy as jnp

from catx.catx import CATX
from catx.network_builder import NetworkBuilder

import numpy as np

from examples.openml_environment import OpenMLEnvironment


def moving_average(x: List[float], w: int) -> np.ndarray:
    return np.convolve(x, np.ones(w), "valid") / w


class MLPBuilder(NetworkBuilder):
    def create_network(self, depth: int) -> hk.Module:
        return hk.nets.MLP([5, 5] + [2 ** (depth + 1)], name=f"mlp_depth_{depth}")


@pytest.mark.parametrize("dataset_id_loss", [(41540, 0.20), (42495, 0.08), (197, 0.15)])
def test_catx_convergence(dataset_id_loss: Tuple[int, float]) -> None:
    """Script to make an integration test on CATS convergence."""
    epsilon = 0.05
    dataset_id, loss_convergence_threshold = dataset_id_loss
    rng_key = jax.random.PRNGKey(42)
    catx_key, env_key = jax.random.split(rng_key, num=2)

    builder = MLPBuilder()
    optimizer = optax.adam(learning_rate=0.01)
    catx = CATX(
        rng_key=catx_key,
        network_builder=builder,
        optimizer=optimizer,
        discretization_parameter=8,
        bandwidth=1 / 8,
    )
    batch_size = 10
    environment = OpenMLEnvironment(dataset_id=dataset_id, batch_size=batch_size)
    costs_cumulative = []
    no_iterations = 1000
    for _ in range(no_iterations):
        env_key, subkey = jax.random.split(env_key)
        obs = environment.get_new_observations(env_key)
        if obs is None:
            break

        actions, probabilities = catx.sample(obs=obs, epsilon=epsilon)

        costs = environment.get_costs(key=subkey, obs=obs, actions=actions)

        catx.learn(obs, actions, probabilities, costs)

        costs_cumulative.append(jnp.mean(costs).item())

        if len(costs_cumulative) > 150:
            if np.mean(costs_cumulative[-100:]) < loss_convergence_threshold:
                break

    costs_cumulative_moving_average = moving_average(costs_cumulative, 100)
    assert costs_cumulative_moving_average[-1] < loss_convergence_threshold
