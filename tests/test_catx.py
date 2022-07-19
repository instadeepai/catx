from typing import List
from unittest.mock import MagicMock

import chex
import jax
import numpy as np
import pytest
import optax
from chex import ArrayNumpy, Array
from pytest_mock import MockerFixture
from jax import numpy as jnp

from catx.catx import CATX
from catx.network_builder import NetworkBuilder

from catx.type_defs import Observations, Actions, Probabilities, JaxObservations


@pytest.fixture
def catx(mlp_builder: NetworkBuilder, request: pytest.FixtureRequest = None) -> CATX:
    if not request:
        action_min = 0.0
        action_max = 1.0
    else:
        action_min = request.param[0]
        action_max = request.param[1]

    rng_key = jax.random.PRNGKey(42)
    rng_key, catx_key = jax.random.split(rng_key, num=2)

    optimizer = optax.adam(learning_rate=0.01)
    catx = CATX(
        rng_key=catx_key,
        network_builder=mlp_builder,
        optimizer=optimizer,
        discretization_parameter=4,
        bandwidth=1.5 / 4,
        action_min=action_min,
        action_max=action_max,
    )

    return catx


@pytest.fixture
def create_forward_fn_mock(mocker: MockerFixture) -> MagicMock:
    mk = mocker.patch("catx.catx.CATX._create_forward_fn")
    return mk  # type: ignore


@pytest.fixture
def create_forward_single_depth_fns_mock(mocker: MockerFixture) -> MagicMock:
    mk = mocker.patch("catx.catx.CATX._create_forward_single_depth_fns")
    mk.return_value = MagicMock(), MagicMock()

    return mk  # type: ignore


@pytest.fixture
def update_mock(mocker: MockerFixture) -> MagicMock:
    mk = mocker.patch("catx.catx.CATX._update")
    mk.return_value = MagicMock(), MagicMock(), MagicMock()
    return mk  # type: ignore


@pytest.fixture
def init_mock(mocker: MockerFixture) -> MagicMock:
    return mocker.patch("catx.catx.CATX._init")  # type: ignore


@pytest.fixture
def init_opt_states_mock(mocker: MockerFixture) -> MagicMock:
    mk = mocker.patch("catx.catx.CATX._init_opt_states")
    mk.return_value = MagicMock()
    return mk  # type: ignore


def test_catx__init(
    catx: CATX, jax_observations: JaxObservations, epsilon: float
) -> None:
    # Validate init initialized:
    # _params, _forward_single_depth_fns _forward_fn, _depth_params, and _opt_states attributes.
    assert not hasattr(catx, "_forward_fn")
    assert not hasattr(catx, "_params")
    assert not hasattr(catx, "_forward_single_depth_fns")
    assert not hasattr(catx, "_depth_params")
    assert not hasattr(catx, "_opt_states")
    catx._init(obs=jax_observations, epsilon=epsilon)
    assert hasattr(catx, "_forward_fn")
    assert hasattr(catx, "_params")
    assert hasattr(catx, "_forward_single_depth_fns")
    assert hasattr(catx, "_depth_params")
    assert hasattr(catx, "_opt_states")


def test_catx__sample(catx: CATX, observations: Observations, epsilon: float) -> None:
    actions, probabilities = catx.sample(obs=observations, epsilon=epsilon)

    # Validate sample return type and shape.
    assert isinstance(probabilities, ArrayNumpy)
    assert isinstance(actions, ArrayNumpy)
    chex.assert_equal_shape([actions, probabilities, observations[:, 0]])

    # Validate stochasticity in sampling.
    actions_, _ = catx.sample(obs=observations, epsilon=epsilon)
    assert not np.all(actions_ == actions)


@pytest.mark.parametrize(
    "catx",
    [(0.0, 1.0), (50.0, 100.0), (-80.0, 20.0), (-60.0, -50.0)],
    indirect=["catx"],
)
def test_catx__sample_action_range(
    catx: CATX, observations: Observations, epsilon: float
) -> None:
    actions, probabilities = catx.sample(obs=observations, epsilon=epsilon)
    assert np.all(catx._action_min <= actions)
    assert np.all(actions <= catx._action_max)


@pytest.mark.parametrize(
    "action_min, action_max", [(0.0, 0.0), (1.0, 0.0), (-5.0, -10.0)]
)
def test_catx__init_action_range_sad(
    mlp_builder: NetworkBuilder, action_min: float, action_max: float
) -> None:
    rng_key = jax.random.PRNGKey(42)
    rng_key, catx_key = jax.random.split(rng_key, num=2)
    optimizer = optax.adam(learning_rate=0.01)
    with pytest.raises(
        AssertionError,
        match=f"'action_max' must be strictly larger than 'action_min', "
        f"got: action_max={action_max} and action_min={action_min}.",
    ):
        CATX(
            rng_key=catx_key,
            network_builder=mlp_builder,
            optimizer=optimizer,
            discretization_parameter=4,
            bandwidth=1.5 / 4,
            action_min=action_min,
            action_max=action_max,
        )


def test_catx__sample_function_call_count(
    catx: CATX,
    create_forward_fn_mock: MagicMock,
    create_forward_single_depth_fns_mock: MagicMock,
    observations: Observations,
    epsilon: float,
) -> None:
    forward_fn_mock = MagicMock()
    forward_fn_mock.return_value = MagicMock(), MagicMock()
    params_mock = MagicMock()
    create_forward_fn_mock.return_value = params_mock, forward_fn_mock

    sample_call_cnt = 3
    for _ in range(sample_call_cnt):
        _, _ = catx.sample(obs=observations, epsilon=epsilon)

    assert create_forward_fn_mock.call_count == 1
    assert create_forward_single_depth_fns_mock.call_count == 1
    assert forward_fn_mock.call_count == sample_call_cnt


def test_catx__learn_calls(
    catx: CATX,
    observations: Observations,
    actions: Actions,
    probabilities: Probabilities,
    costs: ArrayNumpy,
    update_mock: MagicMock,
    init_opt_states_mock: MagicMock,
    epsilon: float,
) -> None:
    learn_call_cnt = 3
    for _ in range(learn_call_cnt):
        catx.learn(
            obs=observations, actions=actions, probabilities=probabilities, costs=costs
        )
    assert init_opt_states_mock.call_count == 1
    assert update_mock.call_count == learn_call_cnt


def test_catx__learn_param_update(
    catx: CATX,
    observations: Observations,
    actions: Actions,
    probabilities: Probabilities,
    costs: ArrayNumpy,
    epsilon: float,
) -> None:
    catx._init(jnp.asarray(observations), epsilon=epsilon)
    rng_pre = catx.rng_key
    depth_params_pre = catx._depth_params
    opt_states_pre = catx._opt_states
    params_pre = catx._params

    catx.learn(
        obs=observations, actions=actions, probabilities=probabilities, costs=costs
    )

    rng_post = catx.rng_key
    depth_params_post = catx._depth_params
    opt_states_post = catx._opt_states
    params_post = catx._params

    assert not jnp.all(rng_pre == rng_post)
    assert not jax.tree_leaves(opt_states_pre) == jax.tree_leaves(opt_states_post)
    assert_not_equal(jax.tree_leaves(params_pre), jax.tree_leaves(params_post))
    assert_not_equal(
        jax.tree_leaves(depth_params_pre), jax.tree_leaves(depth_params_post)
    )


def assert_not_equal(x1: List[Array], x2: List[Array]) -> None:
    f = lambda a, b: np.all(a == b)
    assert not np.all(list(map(f, x1, x2)))
    assert True
