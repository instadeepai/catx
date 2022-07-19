from typing import Dict

import chex
import jax
import haiku as hk
import jax.numpy as jnp
import pytest
import numpy as np
from chex import assert_type

from catx.network_builder import NetworkBuilder
from catx.tree import TreeParameters, Tree
from catx.type_defs import JaxObservations, Logits


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
        match="discretization_parameter must be power of 2 number and larger than 1.",
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
    mlp_builder: NetworkBuilder,
    tree_parameters: TreeParameters,
    jax_observations: JaxObservations,
) -> None:
    def _forward(x: JaxObservations) -> Dict[int, Logits]:
        tree = Tree(
            network_builder=mlp_builder,
            tree_params=tree_parameters,
        )

        output_logits = tree(x)

        # Validate the tree has as many neural networks as depth.
        assert jnp.shape(jax.tree_leaves(tree.networks))[0] == tree_parameters.depth

        return output_logits

    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)

    forward = hk.transform(_forward)

    _params = forward.init(rng=subkey, x=jax_observations)

    # Validate tree initial params.
    chex.assert_tree_all_finite(_params)

    # Query the tree.
    _forward_fn = jax.jit(forward.apply)
    key, subkey = jax.random.split(key)
    logits = _forward_fn(_params, subkey, jax_observations)

    # Validate the network outputs at each depth.
    logits_shape = jax.tree_map(jnp.shape, logits)
    assert len(logits_shape) == tree_parameters.depth
    for d in range(tree_parameters.depth):
        assert logits_shape[d] == (jnp.shape(jax_observations)[0], 2**d, 2)

    chex.assert_tree_all_finite(logits)
