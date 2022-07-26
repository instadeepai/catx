import time
from typing import List
import haiku as hk
import jax
import optax
from jax import numpy as jnp
from catx.catx import CATX
from catx.network_builder import NetworkBuilder
import numpy as np
import matplotlib.pyplot as plt
from examples.openml_environment import OpenMLEnvironment


def moving_average(x: List[float], w: int) -> np.ndarray:
    return np.convolve(x, np.ones(w), "valid") / w


class MLPBuilder(NetworkBuilder):
    def create_network(self, depth: int) -> hk.Module:
        return hk.nets.MLP([10, 10] + [2 ** (depth + 1)], name=f"mlp_depth_{depth}")


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
        network_builder=MLPBuilder(),
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
            state_extras = {"dropout_rate": 0.1}
            state = catx.init(
                obs=obs, epsilon=epsilon, key=key, state_extras=state_extras
            )

        state.state_extras["dropout_rate"] = 0.2
        actions, probabilities, state = catx.sample(
            obs=obs, epsilon=epsilon, state=state
        )

        costs = environment.get_costs(actions=actions)

        state.state_extras["dropout_rate"] = 0.0
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
