from unittest.mock import patch

import jax

from catx.network_builder import NetworkBuilder
import haiku as hk


@patch("catx.network_builder.NetworkBuilder.__abstractmethods__", set())
def test_network_builder() -> None:
    network_builder = NetworkBuilder()  # type: ignore
    assert "create_network" in dir(network_builder)
    assert network_builder.create_network.__isabstractmethod__  # type: ignore


def test_network_builder__create_network() -> None:
    # Sanity check
    class MLPBuilder(NetworkBuilder):
        def create_network(self, depth: int) -> hk.Module:
            return hk.nets.MLP([3] + [2 ** (depth + 1)], name=f"mlp_depth_{depth}")

    class BuilderMocker(hk.Module):
        def __init__(self, network_builder: NetworkBuilder) -> None:
            super().__init__("network_mocker_built")
            self.networks = network_builder.create_network(depth=2)

        def __call__(self) -> int:
            return 1

    def _forward() -> None:
        BuilderMocker(network_builder=builder)

    builder = MLPBuilder()

    forward = hk.transform(_forward)
    rng_key, subkey = jax.random.split(jax.random.PRNGKey(42))
    _ = forward.init(rng=subkey)
