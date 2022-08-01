from typing import List, Tuple, Type

import chex
import jax
import numpy as np
import optax
import pytest
from chex import Array, ArrayNumpy, PRNGKey
from jax import numpy as jnp
from pytest_mock import MockerFixture
from unittest.mock import MagicMock

from catx.catx import CATX, CATXState
from catx.network_module import CATXHaikuNetwork
from catx.type_defs import (
    Actions,
    JaxActions,
    JaxObservations,
    JaxProbabilities,
    Observations,
    Probabilities,
)


def instantiate_catx(
    action_min: float, action_max: float, catx_network: Type[CATXHaikuNetwork]
) -> CATX:
    return CATX(
        catx_network=catx_network,
        optimizer=optax.adam(learning_rate=0.01),
        discretization_parameter=4,
        bandwidth=1.5 / 4,
        action_min=action_min,
        action_max=action_max,
    )


@pytest.fixture
def catx(
    catx_network_without_extras: Type[CATXHaikuNetwork],
    request: pytest.FixtureRequest = None,
) -> CATX:
    if not request:
        action_min = 0.0
        action_max = 1.0
    else:
        action_min = request.param[0]
        action_max = request.param[1]

    return instantiate_catx(action_min, action_max, catx_network_without_extras)


@pytest.fixture
def catx_init(
    catx: CATX,
    jax_observations: JaxObservations,
    epsilon: float,
    key: PRNGKey,
) -> Tuple[CATX, CATXState]:

    state = catx.init(obs=jax_observations, epsilon=epsilon, key=key)

    return catx, state


@pytest.fixture
def catx_with_dropout_extras(
    catx_network_with_dropout_extras: Type[CATXHaikuNetwork],
    request: pytest.FixtureRequest = None,
) -> CATX:
    if not request:
        action_min = 0.0
        action_max = 1.0
    else:
        action_min = request.param[0]
        action_max = request.param[1]

    return instantiate_catx(action_min, action_max, catx_network_with_dropout_extras)


@pytest.fixture
def catx_init_with_extras(
    catx_with_dropout_extras: CATX,
    jax_observations: JaxObservations,
    epsilon: float,
    key: PRNGKey,
) -> Tuple[CATX, CATXState]:

    network_extras = {"dropout_rate": 0.2}
    state = catx_with_dropout_extras.init(
        obs=jax_observations, epsilon=epsilon, key=key, network_extras=network_extras
    )

    return catx_with_dropout_extras, state


@pytest.fixture
def create_forward_fn_mock(mocker: MockerFixture) -> MagicMock:
    return mocker.patch("catx.catx.CATX._create_forward_fn")  # type: ignore


@pytest.fixture
def create_forward_single_depth_fns_mock(mocker: MockerFixture) -> MagicMock:
    mk = mocker.patch("catx.catx.CATX._create_forward_single_depth_fns")
    mk.return_value = MagicMock(), MagicMock()

    return mk  # type: ignore


@pytest.fixture
def init_opt_states_mock(mocker: MockerFixture) -> MagicMock:
    mk = mocker.patch("catx.catx.CATX._init_opt_states")
    mk.return_value = MagicMock()

    return mk  # type: ignore


def test_catx__init(
    catx: CATX,
    jax_observations: JaxObservations,
    epsilon: float,
    key: PRNGKey,
    create_forward_fn_mock: MagicMock,
    create_forward_single_depth_fns_mock: MagicMock,
    init_opt_states_mock: MagicMock,
) -> None:
    # Validate init properly initializes catx.
    create_forward_fn_mock.return_value = MagicMock(), MagicMock()
    assert not hasattr(catx, "_forward_fn")
    assert not hasattr(catx, "_forward_single_depth_fns")
    assert not catx._is_initialized
    state = catx.init(obs=jax_observations, epsilon=epsilon, key=key)
    assert type(state) is CATXState
    assert hasattr(catx, "_forward_fn")
    assert hasattr(catx, "_forward_single_depth_fns")
    assert catx._is_initialized
    assert create_forward_fn_mock.call_count == 1
    assert create_forward_single_depth_fns_mock.call_count == 1
    assert init_opt_states_mock.call_count == 1


def test_catx__init_with_extras(
    catx_with_dropout_extras: CATX,
    jax_observations: JaxObservations,
    epsilon: float,
    key: PRNGKey,
    create_forward_fn_mock: MagicMock,
    create_forward_single_depth_fns_mock: MagicMock,
    init_opt_states_mock: MagicMock,
) -> None:
    # Validate init properly initializes catx with network extras.
    create_forward_fn_mock.return_value = MagicMock(), MagicMock()
    assert not hasattr(catx_with_dropout_extras, "_forward_fn")
    assert not hasattr(catx_with_dropout_extras, "_forward_single_depth_fns")
    assert not catx_with_dropout_extras._is_initialized
    state = catx_with_dropout_extras.init(
        obs=jax_observations,
        epsilon=epsilon,
        key=key,
        network_extras={"dropout_rate": 0.2},
    )
    assert type(state) is CATXState
    assert hasattr(catx_with_dropout_extras, "_forward_fn")
    assert hasattr(catx_with_dropout_extras, "_forward_single_depth_fns")
    assert catx_with_dropout_extras._is_initialized
    assert create_forward_fn_mock.call_count == 1
    assert create_forward_single_depth_fns_mock.call_count == 1
    assert init_opt_states_mock.call_count == 1


@pytest.mark.parametrize("catx_with_init", ["catx_init", "catx_init_with_extras"])
def test_catx__sample(
    catx_with_init: str,
    request: pytest.FixtureRequest,
    observations: Observations,
    epsilon: float,
) -> None:
    catx_with_init = request.getfixturevalue(catx_with_init)
    catx, state = catx_with_init
    actions, probabilities, state = catx.sample(
        obs=observations, epsilon=epsilon, state=state
    )

    # Validate sample return type and shape.
    assert isinstance(probabilities, JaxProbabilities)  # type: ignore
    assert isinstance(actions, JaxActions)  # type: ignore
    chex.assert_equal_shape([actions, probabilities, observations[:, 0]])

    # Validate stochasticity in sampling.
    actions_, _, _ = catx.sample(obs=observations, epsilon=epsilon, state=state)
    assert not np.all(actions_ == actions)


@pytest.mark.parametrize(
    "catx_init",
    [(0.0, 1.0), (50.0, 100.0), (-80.0, 20.0), (-60.0, -50.0)],
    indirect=["catx_init"],
)
def test_catx__sample_action_range(
    catx_init: Tuple[CATX, CATXState], observations: Observations, epsilon: float
) -> None:
    catx, state = catx_init
    actions, probabilities, state = catx.sample(
        obs=observations, epsilon=epsilon, state=state
    )
    assert np.all(catx._action_min <= actions)
    assert np.all(actions <= catx._action_max)


@pytest.mark.parametrize(
    "action_min, action_max", [(0.0, 0.0), (1.0, 0.0), (-5.0, -10.0)]
)
def test_catx__init_action_range_sad(
    catx_network_without_extras: Type[CATXHaikuNetwork],
    action_min: float,
    action_max: float,
) -> None:
    with pytest.raises(
        AssertionError,
        match=f"'action_max' must be strictly larger than 'action_min', "
        f"got: action_max={action_max} and action_min={action_min}.",
    ):
        _ = CATX(
            catx_network=catx_network_without_extras,
            optimizer=optax.adam(learning_rate=0.01),
            discretization_parameter=4,
            bandwidth=1.5 / 4,
            action_min=action_min,
            action_max=action_max,
        )


def test_catx__learn_param_update(
    catx_init: Tuple[CATX, CATXState],
    observations: Observations,
    actions: Actions,
    probabilities: Probabilities,
    costs: ArrayNumpy,
    epsilon: float,
) -> None:
    catx, state = catx_init
    key_pre = state.key
    depth_params_pre = state.depth_params
    opt_states_pre = state.opt_states
    params_pre = state.params

    state = catx.learn(
        obs=observations,
        actions=actions,
        probabilities=probabilities,
        costs=costs,
        state=state,
    )

    key_post = state.key
    depth_params_post = state.depth_params
    opt_states_post = state.opt_states
    params_post = state.params

    assert not jnp.all(key_pre == key_post)
    assert not jax.tree_leaves(opt_states_pre) == jax.tree_leaves(opt_states_post)
    assert_not_equal(jax.tree_leaves(params_pre), jax.tree_leaves(params_post))
    assert_not_equal(
        jax.tree_leaves(depth_params_pre), jax.tree_leaves(depth_params_post)
    )


def assert_not_equal(x1: List[Array], x2: List[Array]) -> None:
    f = lambda a, b: np.all(a == b)
    assert not np.all(list(map(f, x1, x2)))
    assert True
