from typing import List, Optional, Tuple

import haiku as hk
import jax
import numpy as np
import optax
import pytest
import tensorflow as tf
from jax import numpy as jnp
from sklearn.datasets import fetch_openml

from catx.catx import CATX
from catx.network_module import CATXHaikuNetwork
from catx.type_defs import Actions, Costs, Logits, NetworkExtras, Observations


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


class MyCATXNetwork(CATXHaikuNetwork):
    def __init__(self, depth: int) -> None:
        super().__init__(depth)
        self.network = hk.nets.MLP(
            [10, 10] + [2 ** (self.depth + 1)], name=f"mlp_depth_{self.depth}"
        )

    def __call__(
        self,
        obs: Observations,
        network_extras: NetworkExtras,
    ) -> Logits:
        return self.network(
            obs, dropout_rate=network_extras["dropout_rate"], rng=hk.next_rng_key()
        )


@pytest.mark.parametrize("dataset_id_loss", [(41540, 0.20), (42495, 0.08), (197, 0.15)])
def test_catx_convergence(dataset_id_loss: Tuple[int, float]) -> None:
    """Script to make an integration test on CATS convergence."""
    epsilon = 0.05
    dataset_id, loss_convergence_threshold = dataset_id_loss
    rng_key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(rng_key)

    batch_size = 10
    environment = OpenMLEnvironment(dataset_id=dataset_id, batch_size=batch_size)

    catx = CATX(
        catx_network=MyCATXNetwork,
        optimizer=optax.adam(learning_rate=0.01),
        discretization_parameter=8,
        bandwidth=1 / 8,
    )

    costs_cumulative = []
    no_iterations = 1000
    for i in range(no_iterations):
        obs = environment.get_new_observations()
        if obs is None:
            break

        if i == 0:
            network_extras = {"dropout_rate": 0.0}
            state = catx.init(
                obs=obs, epsilon=epsilon, key=key, network_extras=network_extras
            )

        state.network_extras["dropout_rate"] = 0.0
        actions, probabilities, state = catx.sample(
            obs=obs, epsilon=epsilon, state=state
        )

        costs = environment.get_costs(actions=actions)

        state.network_extras["dropout_rate"] = 0.2
        state = catx.learn(
            obs=obs,
            actions=actions,
            probabilities=probabilities,
            costs=costs,
            state=state,
        )

        costs_cumulative.append(jnp.mean(costs).item())

        if len(costs_cumulative) > 150:
            if np.mean(costs_cumulative[-100:]) < loss_convergence_threshold:
                break

    costs_cumulative_moving_average = moving_average(costs_cumulative, 100)
    assert costs_cumulative_moving_average[-1] < loss_convergence_threshold
