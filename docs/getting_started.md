# Getting started:
This example uses
the [black Friday dataset from OpenML](https://www.openml.org/search?type=data&status=active&id=41540).

The task is to predict how much an individual will purchase, i.e., continuous action,
based on the feature of that individual, i.e., context.



## Environment class
For simplicity, we normalize the feature and action spaces.
```python
from typing import Optional
import tensorflow as tf
from sklearn.datasets import fetch_openml
import numpy as np
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
In this example, we use a multilayer perceptron (MLP) network with dropouts.
> **_IMPORTANT:_**  The number of neurons at the output layer should be 2**(depth+1)

```python
# CATX imports
import time
from typing import List
import haiku as hk
import jax
from chex import PRNGKey
import optax
from jax import numpy as jnp
from catx.catx import CATX
from catx.network_module import CustomHaikuNetwork
import numpy as np
import matplotlib.pyplot as plt
from catx.type_defs import Observations, NetworkExtras, Logits


# Network builder
class MyCustomNetwork(CustomHaikuNetwork):
    def __init__(self, depth: int) -> None:
        super().__init__(depth)
        self.network = hk.nets.MLP(
            [10, 10] + [2 ** (self.depth + 1)], name=f"mlp_depth_{self.depth}"
        )

    def __call__(
        self,
        obs: Observations,
        key: PRNGKey,
        network_extras: NetworkExtras,
    ) -> Logits:
        return self.network(obs, dropout_rate=network_extras["dropout_rate"], rng=key)


def moving_average(x: List[float], w: int) -> np.ndarray:
    return np.convolve(x, np.ones(w), "valid") / w


def main() -> None:
    start_time = time.time()
    epsilon = 0.05

    # JAX pseudo-random number generator
    rng_key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(rng_key)

    # Instantiate the environment
    environment = BlackFridayEnvironment()

    # Instantiate CATX
    catx = CATX(
        custom_network=MyCustomNetwork,
        optimizer=optax.adam(learning_rate=0.01),
        discretization_parameter=8,
        bandwidth=1 / 8,
    )

    # Training loop
    costs_cumulative = []
    for i in range(1000):
        obs = environment.get_new_observations()
        if obs is None:
            break

        if i == 0:
            network_extras = {"dropout_rate": 0.2}
            state = catx.init(
                obs=obs, epsilon=epsilon, key=key, network_extras=network_extras
            )

        state.network_extras["dropout_rate"] = 0.2
        actions, probabilities, state = catx.sample(
            obs=obs, epsilon=epsilon, state=state
        )
        actions = np.array(actions)
        probabilities = np.array(probabilities)

        costs = environment.get_costs(actions=actions)

        state.network_extras["dropout_rate"] = 0.0
        state = catx.learn(
            obs=obs,
            actions=actions,
            probabilities=probabilities,
            costs=costs,
            state=state,
        )

        costs_cumulative.append(jnp.mean(costs).item())

    plt.plot(costs_cumulative)
    plt.plot(moving_average(costs_cumulative, 50))
    plt.title("Action costs")
    plt.show()

    print(f"CATX training took {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    main()
```
