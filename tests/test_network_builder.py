from unittest.mock import patch
from chex import PRNGKey
from catx.network_module import CustomHaikuNetwork
import haiku as hk


@patch("catx.network_module.CustomHaikuNetwork.__abstractmethods__", set())
def test_network_module(key: PRNGKey) -> None:
    def _forward() -> None:
        netowrk = CustomHaikuNetwork(depth=2)  # type: ignore
        assert hasattr(netowrk, "depth")

    forward = hk.transform(_forward)
    params = forward.init(rng=key)
    forward.apply(params, key)

    assert "__call__" in dir(CustomHaikuNetwork)
    assert CustomHaikuNetwork.__call__.__isabstractmethod__  # type: ignore
    assert issubclass(CustomHaikuNetwork, hk.Module)
