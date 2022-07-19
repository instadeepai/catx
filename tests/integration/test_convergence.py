from typing import List, Tuple, Optional
import jax
import optax
import pytest
from jax import numpy as jnp
import haiku as hk
import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_openml
from catx.catx import CATX
from catx.network_builder import NetworkBuilder
from catx.type_defs import Actions, Costs, Observations


def moving_average(x: List[float], w: int) -> np.ndarray:
    return np.convolve(x, np.ones(w), "valid") / w


class OpenMLEnvironment:
    def __init__(self, dataset_id: int, batch_size: int = 5) -> None:
        self.x, self.y = fetch_openml(
            data_id=dataset_id, as_frame=False, return_X_y=True
        )
        rows_with_nan_idx = np.argwhere(np.isnan(self.x))[:, 0]
        self.x = np.delete(self.x, rows_with_nan_idx, axis=0)
        self.y = np.delete(self.y, rows_with_nan_idx, axis=0)
        self.x = self._normalize_data(self.x)
        self.y = self._normalize_data(self.y)
        self._y_mean = np.mean(self.y)
        physical_devices = tf.config.list_physical_devices("GPU")

        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except Exception:
            pass

        self.dataset = tf.data.Dataset.from_tensor_slices((self.x, self.y))
        self.dataset = self.dataset.batch(batch_size)
        self.iterator = iter(self.dataset)

    def get_new_observations(self) -> Optional[Observations]:
        try:
            x, y = self.iterator.get_next()
            self.x = x.numpy()
            self.y = y.numpy()
            return self.x
        except tf.errors.OutOfRangeError:
            return None

    def get_costs(self, actions: Actions) -> Costs:
        costs = np.abs(actions - self.y)
        return costs

    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        return (data - np.min(data, axis=0)) / (
            np.max(data, axis=0) - np.min(data, axis=0)
        )


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
        obs = environment.get_new_observations()
        if obs is None:
            break

        actions, probabilities = catx.sample(obs=obs, epsilon=epsilon)

        costs = environment.get_costs(actions=actions)

        catx.learn(obs, actions, probabilities, costs)

        costs_cumulative.append(jnp.mean(costs).item())

        if len(costs_cumulative) > 150:
            if np.mean(costs_cumulative[-100:]) < loss_convergence_threshold:
                break

    costs_cumulative_moving_average = moving_average(costs_cumulative, 100)
    assert costs_cumulative_moving_average[-1] < loss_convergence_threshold
