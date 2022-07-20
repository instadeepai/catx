# CATX code example:
This example uses
the [black Friday dataset from OpenML](https://www.openml.org/search?type=data&status=active&id=41540).

The task is to predict how much an individual will purchase, i.e., continuous action,
based on the feature of that individual, i.e., context.


## Environment class
For simplicity, we normalize the feature and action spaces.
```python
from typing import Optional
import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_openml
from catx.type_defs import Actions, Costs, Observations

class BlackFridayEnvironment:
    def __init__(self, batch_size: int = 10) -> None:
        self.x, self.y = fetch_openml(
            data_id=41540, as_frame=False, return_X_y=True
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
```

## Training loop
One of the main advantages of CATX is the custom network builder.
In this example, we use a multilayer perceptron (MLP) network builder.
> **_IMPORTANT:_**  The number of neurons at the output layer should be should be 2**(depth+1)

```python
# CATX imports
import haiku as hk
import jax
import optax
from catx.catx import CATX
from catx.network_builder import NetworkBuilder

# Network builder
class MLPBuilder(NetworkBuilder):
    def create_network(self, depth: int) -> hk.Module:
        return hk.nets.MLP([10, 10] + [2 ** (depth + 1)], name=f"mlp_depth_{depth}")


def main() -> None:
    # JAX pseudo-random number generator
    rng_key = jax.random.PRNGKey(42)
    catx_key, env_key = jax.random.split(rng_key, num=2)

    # Instantiate CATX
    builder = MLPBuilder()
    optimizer = optax.adam(learning_rate=0.01)
    catx = CATX(
        rng_key=catx_key,
        network_builder=builder,
        optimizer=optimizer,
        discretization_parameter=8,
        bandwidth=1 / 8,
    )

    # Instantiate the environment (black_friday dataset id from https://www.openml.org/)
    environment = BlackFridayEnvironment()

    # Training loop
    for _ in range(1000):
        env_key, cost_key = jax.random.split(env_key)
        obs = environment.get_new_observations(env_key)
        if obs is None:
            break
        actions, probabilities = catx.sample(obs=obs, epsilon=0.05)
        costs = environment.get_costs(key=cost_key, obs=obs, actions=actions)
        catx.learn(obs, actions, probabilities, costs)


if __name__ == "__main__":
    main()
```
