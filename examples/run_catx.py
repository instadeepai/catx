from typing import List

import haiku as hk
import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
import time
from jax import numpy as jnp

from catx.catx import CATX
from catx.network_module import CATXHaikuNetwork
from catx.type_defs import Logits, NetworkExtras, Observations
from examples.openml_environment import OpenMLEnvironment


def moving_average(x: List[float], w: int) -> np.ndarray:
    return np.convolve(x, np.ones(w), "valid") / w


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


def main() -> None:
    print('Running CATX training on the "black_friday" dataset from OpenML.')
    dataset_id = 41540  # black_friday dataset id from OpenML
    epsilon = 0.05
    start_time = time.time()

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
    for i in range(1000):
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

    plt.plot(costs_cumulative)
    plt.plot(moving_average(costs_cumulative, 50))
    plt.title("Action costs")
    plt.show()

    print(f"CATX training took {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
