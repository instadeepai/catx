from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple, Type

import chex
import functools
import haiku as hk
import jax
import optax
from chex import Array
from jax import numpy as jnp
from jax.stages import Wrapped

from catx.network_module import CATXHaikuNetwork
from catx.tree import Tree, TreeParameters
from catx.type_defs import (
    Actions,
    Costs,
    JaxActions,
    JaxCosts,
    JaxLoss,
    JaxObservations,
    JaxProbabilities,
    Logits,
    NetworkExtras,
    Observations,
    Probabilities,
)

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass


@dataclass
class CATXState:
    """Holds the CATX's training state and extra parameterization for the networks."""

    params: hk.Params
    depth_params: Dict[int, hk.Params]
    opt_states: optax.OptState
    key: chex.PRNGKey
    network_extras: NetworkExtras


class CATX:
    """This class runs CATX action sampling and training of the tree."""

    def __init__(
        self,
        catx_network: Type[CATXHaikuNetwork],
        optimizer: optax.GradientTransformation,
        discretization_parameter: int,
        bandwidth: float,
        action_min: float = 0.0,
        action_max: float = 1.0,
    ):
        """Instantiate a CATX instance with its corresponding tree.

        Args:
            catx_network: class specifying the neural network architecture.
            optimizer: optax optimizer object.
            discretization_parameter: the number of action centroids.
            bandwidth: the bucket half width covered by action centroid.
            action_min: the lowest value of the action space.
            action_max: the highest value of the action space.
        """

        assert action_max > action_min, (
            f"'action_max' must be strictly larger than 'action_min', "
            f"got: action_max={action_max} and action_min={action_min}."
        )

        self.discretization_parameter = discretization_parameter
        self.bandwidth = bandwidth
        self.catx_network = catx_network
        self.optimizer = optimizer
        self.tree_params = TreeParameters.construct(
            bandwidth=bandwidth, discretization_parameter=discretization_parameter
        )

        self._action_min = action_min
        self._action_max = action_max

        self._is_initialized = False

        self._forward_fn: Wrapped
        self._forward_single_depth_fns: Dict[int, Wrapped]

    @functools.partial(jax.jit, static_argnames=("self",))
    def sample(
        self,
        obs: Observations,
        epsilon: float,
        state: CATXState,
    ) -> Tuple[Actions, Probabilities, CATXState]:
        """Samples an action from the tree.

        Args:
            obs: the observations, i.e., batched contexts.
            epsilon: probability of selecting a random action.
            state: holds the CATX's training state.

        Returns:
            actions: sampled actions from the tree using epsilon-greedy.
            probabilities: the probability density value of the sampled actions.
            state: holds the CATX's training state.
        """

        obs = jnp.asarray(obs)

        key, subkey = jax.random.split(state.key)

        actions, probabilities = self._forward_fn(
            params=state.params,
            x=obs,
            rng=subkey,
            epsilon=epsilon,
            network_extras=state.network_extras,
        )

        state_new = CATXState(
            params=state.params,
            depth_params=state.depth_params,
            opt_states=state.opt_states,
            key=key,
            network_extras=state.network_extras,
        )

        return actions, probabilities, state_new

    @functools.partial(jax.jit, static_argnames=("self",))
    def learn(
        self,
        obs: Observations,
        actions: Actions,
        probabilities: Probabilities,
        costs: Costs,
        state: CATXState,
    ) -> CATXState:
        """Updates the tree:
            - updates the parameters of the depth specific neural networks.
            - copy the parameters of the depth specific neural networks to the tree neural networks.
            - update the state of the optimizers.
            - update the pseudo random key generator.

        Args:
            obs: the observations, i.e., batched contexts.
            actions: The executed actions associated with the given observations.
            probabilities: the probability density value of the actions.
            costs: The costs incurred from the executed actions.
            state: holds the CATX's training state.

        Returns:
            state: holds the CATX's training state.

        """

        obs = jnp.asarray(obs)

        key, subkey = jax.random.split(state.key)

        rng_key, new_layers_params, new_opt_states = self._update(
            rng_key=subkey,
            layers_params=state.depth_params,
            opt_states=state.opt_states,
            obs=obs,
            actions=actions,
            probabilities=probabilities,
            costs=costs,
            network_extras=state.network_extras,
        )

        params = {
            k: v
            for layer_params in new_layers_params.values()
            for k, v in layer_params.items()
        }

        state_new = CATXState(
            params=params,
            depth_params=new_layers_params,
            opt_states=new_opt_states,
            key=key,
            network_extras=state.network_extras,
        )

        return state_new

    def init(
        self,
        obs: JaxObservations,
        key: chex.PRNGKey,
        epsilon: float,
        network_extras: Optional[NetworkExtras] = None,
    ) -> CATXState:
        """Initializes the parameters of tree's neural networks,
        the forward functions, and the optimizer states.

        This function can only be called once. It is called the first time a CATX instance is used.

        Args:
            obs: the observations, i.e., batched contexts.
            key: pseudo-random number generator.
            epsilon: probability of selecting a random action.
            network_extras: additional information for querying the neural networks.

        Returns:
            state: holds the CATX's training state.
        """

        if network_extras is None:
            network_extras = {}

        key, key_forward_fn, key_single_depth_fns = jax.random.split(key, num=3)

        params, self._forward_fn = self._create_forward_fn(
            obs=obs,
            epsilon=epsilon,
            key=key_forward_fn,
            network_extras=network_extras,
        )

        (
            self._forward_single_depth_fns,
            depth_params,
        ) = self._create_forward_single_depth_fns(
            obs=obs,
            key=key_single_depth_fns,
            network_extras=network_extras,
        )

        opt_states = self._init_opt_states(depth_params)

        state = CATXState(
            params=params,
            depth_params=depth_params,
            opt_states=opt_states,
            key=key,
            network_extras=network_extras,
        )

        self._is_initialized = True

        return state

    def _create_forward_fn(
        self,
        obs: Observations,
        epsilon: float,
        key: chex.PRNGKey,
        network_extras: NetworkExtras,
    ) -> Tuple[hk.Params, Wrapped]:
        """Creates a jitted forward function of the tree
        and initializes the parameters of the tree's neural networks.

        Args:
            obs: the observations, i.e., batched contexts.
            epsilon: probability of selecting a random action.
            key: pseudo-random number generator.
            network_extras: additional information for querying the neural networks.

        Returns:
            _params: the parameters of the neural networks
            _forward_fn: a jitted forward function of the tree.
        """

        def _forward(
            x: JaxObservations,
            epsilon: float,
            network_extras: NetworkExtras,
        ) -> Tuple[JaxActions, JaxProbabilities]:
            """This forward function defines how the tree is traversed and how actions are sampled:
                - All the tree logits are queried (one set of pairwise logits per tree depth).
                - The tree is traversed by following the max of the logits at each
                  tree depth until an action centroid is reached.
                - With a probability 1-epsilon an action is sampled uniformly from
                  the centroid action space and with a probability epsilon an action
                  is uniformly sampled from action space.

            Args:
                x: the observations, i.e., batched contexts.
                epsilon: probability of selecting a random action.
                network_extras: additional information for querying the neural networks.

            Returns:
                actions: sampled actions from the tree using epsilon-greedy
                        and scaled to the environment action range.
                probabilities: the probability density value of the actions.
            """

            tree = Tree(
                catx_network=self.catx_network,
                tree_params=self.tree_params,
            )

            (
                key_exploration,
                key_exploitation,
                key_sampling_exploration,
            ) = jax.random.split(hk.next_rng_key(), num=3)

            logits = tree(obs=x, network_extras=network_extras)

            batch_size = x.shape[0]
            mask = logits[0] < jnp.max(logits[0], axis=(1, 2)).reshape(batch_size, 1, 1)

            # Traverse the tree from root to action leaf
            d = tree.tree_params.depth - 1
            for i in range(1, d):
                c = jnp.where(
                    mask.reshape(logits[i].shape[0:2] + (1,)), -jnp.inf, logits[i]
                )
                mask = c < jnp.max(c, axis=(1, 2)).reshape(batch_size, 1, 1)

            c = jnp.where(
                mask.reshape(logits[d].shape[0:2] + (1,)), -jnp.inf, logits[d]
            )
            actions_centroid = jnp.argmax(c.reshape(batch_size, -1), axis=-1)
            probabilities = tree.tree_params.probabilities[actions_centroid]

            # epsilon-greedy
            probabilities = (1 - epsilon) * probabilities + epsilon
            action_spaces = tree.tree_params.spaces[actions_centroid]

            exploitation_actions = jax.random.uniform(
                key_exploitation,
                shape=actions_centroid.shape,
                minval=action_spaces[:, 0],
                maxval=action_spaces[:, 1],
            )

            exploration = jax.random.choice(
                key_exploration,
                a=jnp.asarray([False, True]),
                p=jnp.asarray([1 - epsilon, epsilon]),
                shape=actions_centroid.shape,
            )

            probabilities = jnp.where(exploration, epsilon, probabilities)

            actions = jnp.where(
                exploration,
                jax.random.uniform(
                    key_sampling_exploration, shape=actions_centroid.shape
                ),
                exploitation_actions,
            )

            # Scale sampled actions to the environment action range.
            actions = actions * (self._action_max - self._action_min) + self._action_min
            probabilities /= self._action_max - self._action_min

            return actions, probabilities

        forward = hk.transform(_forward)

        _params = forward.init(
            rng=key,
            x=obs,
            epsilon=epsilon,
            network_extras=network_extras,
        )
        _forward_fn = jax.jit(forward.apply)

        return _params, _forward_fn

    def _create_forward_single_depth_fns(
        self,
        obs: Observations,
        key: chex.PRNGKey,
        network_extras: NetworkExtras,
    ) -> Tuple[Dict[int, Wrapped], Dict[int, hk.Params]]:
        """Creates a dictionary of jitted forward functions, one per neural network at each tree depth
        and initializes the parameters of these neural networks.

        Args:
            obs: the observations, i.e., batched contexts.
            network_extras: additional information for querying the neural networks.
            key: pseudo-random number generator.

        Returns:
            _forward_single_depth_fns: a dictionary of jitted forward functions of neural networks
                                    with tree depth as key.
            _depth_params: a dictionary of neural network parameters with tree depth as key.
        """

        def create_single_depth_function(
            depth: int,
        ) -> Callable[[JaxObservations, NetworkExtras], Logits]:
            """Creates a neural network forward function for a given depth.

            Args:
                depth: depth at which the neural network will be used.

            Returns:
                _forward: the neural network forward function.
            """

            n_leafs = 2 ** (depth + 1)

            def _forward(x: JaxObservations, network_extras: NetworkExtras) -> Logits:
                """Creates a neural network forward function for a predefined depth.

                Args:
                    x: the observations, i.e., batched contexts.
                    network_extras: additional information for querying the neural networks.

                Returns:
                    the neural network forward function at the predefined depth.
                """

                tree = Tree(
                    catx_network=self.catx_network,
                    tree_params=self.tree_params,
                )
                return tree.networks[depth](
                    obs=x,
                    network_extras=network_extras,
                ).reshape(x.shape[0], n_leafs // 2, 2)

            return _forward

        transformed_layers = {
            i: hk.transform(create_single_depth_function(i))
            for i in range(self.tree_params.depth)
        }
        _forward_single_depth_fns = {
            i: jax.jit(func.apply) for i, func in transformed_layers.items()
        }

        keys = jax.random.split(key, num=self.tree_params.depth)

        _depth_params = {
            i: layer.init(x=obs, rng=keys[i], network_extras=network_extras)
            for i, layer in transformed_layers.items()
        }

        return _forward_single_depth_fns, _depth_params

    def _init_opt_states(
        self, depth_params: Dict[int, hk.Params]
    ) -> Dict[int, optax.OptState]:
        """Initializes an optimizer state for each tree depth
        using the parameters of the depth specific neural networks.

        Returns:
            a dictionary initialized optax optimizer states with tree depth as a key.
        """

        _opt_states = {
            d: self.optimizer.init(params) for d, params in depth_params.items()
        }

        return _opt_states

    @functools.partial(jax.jit, static_argnames=("self",))
    def _compute_smooth_costs(
        self, costs: JaxCosts, actions: JaxActions, probabilities: JaxProbabilities
    ) -> JaxCosts:
        """Computes the smooth cost for a given batch of actions.

        Args:
            costs: the costs incurred from the executed actions.
            actions: the executed actions in the tree action range [0, 1].
            probabilities: the probability density value of the actions.

        Returns:
            the smooth costs
        """

        action_space, volumes = (
            self.tree_params.action_space,
            self.tree_params.volumes,
        )
        indicator = (
            jnp.abs(actions.reshape(-1, 1) - action_space.reshape(1, -1))
            < self.bandwidth
        )

        return (
            costs.reshape(-1, 1)
            * indicator.astype(jnp.float32)
            / (volumes.reshape(1, -1) * probabilities.reshape(-1, 1))
        ).reshape(costs.shape[0], self.tree_params.discretization_parameter // 2, 2)

    @functools.partial(jax.jit, static_argnames=("self", "depth"))
    def _loss(
        self,
        layer_params: hk.Params,
        obs: JaxObservations,
        smooth_costs: JaxCosts,
        depth: int,
        mask_eq: Array,
        rng_key: chex.PRNGKey,
        network_extras: NetworkExtras,
    ) -> JaxLoss:
        """Computes the loss function at a given depth.

        Args:
            layer_params: a dictionary of neural network parameters with tree depth as key.
            obs: the observations, i.e., batched contexts.
            smooth_costs: the smooth costs.
            depth: the tree depth at which the loss will be calculated.
            mask_eq: a mask with zeros where smooth cost pairs are equal
            rng_key: JAX key generator.
            network_extras: additional information for querying the neural networks.

        Returns:
            the sum of the batch losses.
        """

        logits = self._forward_single_depth_fns[depth](
            params=layer_params,
            x=obs,
            rng=rng_key,
            network_extras=network_extras,
        )

        smooth_costs_filtered = jnp.multiply(mask_eq, smooth_costs)
        logits_filtered = jnp.multiply(mask_eq, logits)

        return jnp.sum(jax.nn.softmax(logits_filtered) * smooth_costs_filtered)

    @functools.partial(jax.jit, static_argnames=("self",))
    def _update(
        self,
        rng_key: chex.PRNGKey,
        layers_params: Dict[int, hk.Params],
        opt_states: Dict[int, optax.OptState],
        obs: JaxObservations,
        actions: JaxActions,
        probabilities: JaxProbabilities,
        costs: JaxCosts,
        network_extras: NetworkExtras,
    ) -> Tuple[chex.PRNGKey, Dict[int, hk.Params], Dict[int, optax.OptState]]:
        """Performs the update of the neural networks.

        Args:
            rng_key: JAX key generator.
            layers_params: a dictionary of neural network parameters with tree depth as key.
            opt_states: a dictionary of optimizer states with tree depth as a key.
            obs: the observations, i.e., batched contexts.
            actions: the executed actions in the environment action range.
            probabilities: the probability density value of the actions.
            costs: the costs incurred from the executed actions.
            network_extras: additional information for querying the neural networks.

        Returns:
            rng_key: JAX key generator.
            new_layer_params: a dictionary of the updated network parameters with tree depth as key.
            new_opt_states: a dictionary of updated optimizer states with tree depth as a key.

        """

        # Scale actions from the environment action range to the tree action range.
        actions = (actions - self._action_min) / (self._action_max - self._action_min)
        probabilities *= self._action_max - self._action_min

        smooth_costs = self._compute_smooth_costs(
            costs=costs, actions=actions, probabilities=probabilities
        )

        new_layer_params = {}
        new_opt_states = {}

        for depth in reversed(range(self.tree_params.depth)):
            rng_key, loss_key, layer_key = jax.random.split(rng_key, num=3)
            # Create a mask with zeros where smooth cost pairs are equal
            mask_eq = jnp.logical_not(
                jnp.isclose(smooth_costs[:, :, 0], smooth_costs[:, :, 1])
            ).astype(jnp.int32)

            mask_eq = jnp.expand_dims(mask_eq, axis=-1)
            mask_eq = jnp.tile(mask_eq, reps=2)

            # Compute and apply gradient
            grads = jax.grad(self._loss)(
                layers_params[depth],
                obs,
                smooth_costs,
                depth,
                mask_eq,
                loss_key,
                network_extras,
            )
            updates, opt_state = self.optimizer.update(grads, opt_states[depth])
            new_layer_params[depth] = optax.apply_updates(layers_params[depth], updates)
            new_opt_states[depth] = opt_state

            # Update smooth cost for the next upper tree depth
            if depth > 0:
                # Get smooth cost from 1 step forward following the updated parameters
                logits = self._forward_single_depth_fns[depth](
                    params=new_layer_params[depth],
                    x=obs,
                    rng=layer_key,
                    network_extras=network_extras,
                )
                mask = logits < jnp.max(logits, axis=-1).reshape(
                    logits.shape[0:2] + (1,)
                )
                c = jnp.where(mask, -jnp.inf, smooth_costs)
                smooth_costs = jnp.max(c, axis=-1).reshape(
                    logits.shape[0], logits.shape[1] // 2, 2
                )

        return rng_key, new_layer_params, new_opt_states
