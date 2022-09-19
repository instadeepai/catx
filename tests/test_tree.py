from typing import Dict, Type

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from chex import assert_type

from catx.network_module import CATXHaikuNetwork
from catx.tree import Tree, TreeParameters
from catx.type_defs import JaxObservations, Logits, NetworkExtras


@pytest.mark.parametrize("bandwidth", [1.5 / 4, 1 / 8])
@pytest.mark.parametrize("discretization_parameter", [4, 8, 16])
def test_tree_parameters__creation(
    bandwidth: float, discretization_parameter: int
) -> None:
    tree_params = TreeParameters.construct(
        bandwidth=bandwidth, discretization_parameter=discretization_parameter
    )
    assert hasattr(tree_params, "bandwidth")
    assert hasattr(tree_params, "discretization_parameter")
    assert hasattr(tree_params, "action_space")
    assert hasattr(tree_params, "depth")
    assert hasattr(tree_params, "spaces")
    assert hasattr(tree_params, "volumes")
    assert hasattr(tree_params, "probabilities")

    assert isinstance(tree_params.bandwidth, float)
    assert isinstance(tree_params.discretization_parameter, int)
    assert tree_params.depth >= 1
    assert isinstance(tree_params.depth, int)
    assert_type(tree_params.action_space, float)
    assert_type(tree_params.spaces, float)
    assert_type(tree_params.volumes, float)
    assert_type(tree_params.probabilities, float)

    assert jnp.shape(tree_params.action_space)[0] == discretization_parameter

    assert int(np.log2(tree_params.discretization_parameter)) == tree_params.depth
    assert jnp.min(tree_params.spaces) >= 0
    assert jnp.max(tree_params.spaces) <= 1


@pytest.mark.parametrize("discretization_parameter", [-5, -4, 0, 1, 5, 6, 9, 12])
def test_tree_parameters__discretization_parameter(
    discretization_parameter: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="discretization_parameter must be a power of 2 number and larger than 1.",
    ):
        TreeParameters.construct(
            bandwidth=1 / 8, discretization_parameter=discretization_parameter
        )


@pytest.mark.parametrize("bandwidth", [-5, -1, 0])
def test_tree_parameters__bandwidth(bandwidth: float) -> None:
    with pytest.raises(
        AssertionError, match="Only positive bandwidth value is admissible."
    ):
        TreeParameters.construct(bandwidth=bandwidth, discretization_parameter=4)


def test_tree(
    catx_network_without_extras: Type[CATXHaikuNetwork],
    tree_parameters: TreeParameters,
    jax_observations: JaxObservations,
    epsilon: float,
) -> None:
    def _forward(
        x: JaxObservations,
        network_extras: NetworkExtras,
    ) -> Dict[int, Logits]:
        tree = Tree(
            catx_network=catx_network_without_extras,
            tree_params=tree_parameters,
        )

        output_logits: Dict[int, Logits] = tree(obs=x, network_extras=network_extras)

        # Validate the tree has as many neural networks as depth.
        assert jnp.shape(jax.tree_leaves(tree.networks))[0] == tree_parameters.depth

        return output_logits

    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)

    forward = hk.transform(_forward)

    _params = forward.init(
        rng=key,
        x=jax_observations,
        network_extras={},
    )

    # Validate tree initial params.
    chex.assert_tree_all_finite(_params)

    # Query the tree.
    _forward_fn = jax.jit(forward.apply)
    key, subkey = jax.random.split(key)
    logits = _forward_fn(
        params=_params, x=jax_observations, rng=subkey, network_extras={}
    )

    # Validate the network outputs at each depth.
    logits_shape = jax.tree_map(jnp.shape, logits)
    assert len(logits_shape) == tree_parameters.depth
    for d in range(tree_parameters.depth):
        assert logits_shape[d] == (jnp.shape(jax_observations)[0], 2**d, 2)

    chex.assert_tree_all_finite(logits)


def test_tree_parameters__probabilities_and_volumes() -> None:
    tree_param = TreeParameters.construct(bandwidth=1 / 4, discretization_parameter=4)

    expected_volumes = jnp.full_like(tree_param.volumes, 1 / 2)
    expected_probabilities = jnp.full_like(tree_param.probabilities, 2)

    assert jnp.allclose(tree_param.volumes, expected_volumes)
    assert jnp.allclose(tree_param.probabilities, expected_probabilities)
