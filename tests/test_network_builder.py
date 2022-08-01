import haiku as hk
from chex import PRNGKey
from unittest.mock import patch

from catx.network_module import CATXHaikuNetwork


@patch("catx.network_module.CATXHaikuNetwork.__abstractmethods__", set())
def test_network_module(key: PRNGKey) -> None:
    def _forward() -> None:
        netowrk = CATXHaikuNetwork(depth=2)  # type: ignore
        assert hasattr(netowrk, "depth")

    forward = hk.transform(_forward)
    params = forward.init(rng=key)
    forward.apply(params, key)

    assert "__call__" in dir(CATXHaikuNetwork)
    assert CATXHaikuNetwork.__call__.__isabstractmethod__  # type: ignore
    assert issubclass(CATXHaikuNetwork, hk.Module)
