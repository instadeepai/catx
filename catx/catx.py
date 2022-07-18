import functools
from typing import Callable, Tuple, Dict

import chex
import haiku as hk
import jax
import numpy as np
import optax
from chex import PRNGKey
from jax import numpy as jnp
from chex import Array
from jax.stages import Wrapped

from catx.network_builder import NetworkBuilder
from catx.tree import Tree, TreeParameters
from catx.type_defs import (
    Actions,
    Costs,
    JaxActions,
    JaxCosts,
    JaxObservations,
    JaxProbabilities,
    Logits,
    Observations,
    Probabilities,
    JaxLoss,
)


class CATX:
    """This class runs CATX action sampling and training of the tree."""

    def __init__(
        self,
        rng_key: PRNGKey,
        network_builder: NetworkBuilder,
        optimizer: optax.GradientTransformation,
        discretization_parameter: int,
        bandwidth: float,
        action_min: float = 0.0,
        action_max: float = 1.0,
    ):
        """Instantiate a CATX instance with its corresponding tree.

        Args:
            rng_key: JAX key generator.
            network_builder: specify the neural network architecture for each depth in the tree.
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
        self.network_builder = network_builder
        self.optimizer = optimizer
        self.rng_key = rng_key
        self.tree_params = TreeParameters.construct(
            bandwidth=bandwidth, discretization_parameter=discretization_parameter
        )

        self._action_min = action_min
        self._action_max = action_max

        self.is_initialized = False

        self._forward_fn: Wrapped
        self._forward_single_depth_fns: Dict[int, Wrapped]
        self._params: hk.Params
        self._depth_params: Dict[int, hk.Params]
        self._opt_states: Dict[int, optax.OptState]

    def sample(
        self, obs: Observations, epsilon: float
    ) -> Tuple[Actions, Probabilities]:
        """Samples an action from the tree.

        Args:
            obs: the observations, i.e., batched contexts.
            epsilon: probability of selecting a random action.

        Returns:
            actions: sampled actions from the tree using epsilon-greedy.
            probabilities: the probability density value of the sampled actions.
        """

        obs = jnp.asarray(obs)

        if not self.is_initialized:
            self._init(obs=obs, epsilon=epsilon)

        self.rng_key, sample_key = jax.random.split(self.rng_key, 2)

        actions, probabilities = self._forward_fn(
            self._params, x=obs, epsilon=epsilon, rng=sample_key
        )

        return np.array(actions), np.array(probabilities)

    def learn(
        self,
        obs: Observations,
        actions: Actions,
        probabilities: Probabilities,
        costs: Costs,
    ) -> None:
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
        """

        obs = jnp.asarray(obs)

        if not self.is_initialized:
            self._init(obs=obs)

        rng_key, new_layers_params, new_opt_states = self._update(
            rng_key=self.rng_key,
            layers_params=self._depth_params,
            opt_states=self._opt_states,
            obs=obs,
            actions=actions,
            probabilities=probabilities,
            costs=costs,
        )

        self.rng_key = rng_key
        self._depth_params = new_layers_params
        self._opt_states = new_opt_states

        self._params = {
            k: v
            for layer_params in new_layers_params.values()
            for k, v in layer_params.items()
        }

    def _init(self, obs: JaxObservations, epsilon: float = 0.0) -> None:
        """Initializes the parameters of tree's neural networks,
        the forward functions, and the optimizer states.

        This functions can only be called once. It is called the first time a CATX instance is used.

        Args:
            obs: the observations, i.e., batched contexts.
        """

        if self.is_initialized:
            return

        self.is_initialized = True
        self._params, self._forward_fn = self._create_forward_fn(
            obs=obs, epsilon=epsilon
        )
        (
            self._forward_single_depth_fns,
            self._depth_params,
        ) = self._create_forward_single_depth_fns(obs)

        self._opt_states = self._init_opt_states()

    def _create_forward_fn(
        self, obs: Observations, epsilon: float
    ) -> Tuple[hk.Params, Wrapped]:
        """Creates a jitted forward function of the tree
        and initializes the parameters of tree's neural networks.

        Args:
            obs: the observations, i.e., batched contexts.

        Returns:
            _params: the parameters of the neural networks
            _forward_fn: a jitted forward function of the tree.
        """

        def _forward(
            x: JaxObservations, epsilon: float
        ) -> Tuple[JaxActions, JaxProbabilities]:
            """This forward function defines how the tree is traversed and how actions sampled:
                - All the tree logits are queried (one set of pairwise logits per tree depth).
                - The tree is traversed by following the max of the logits at each
                  tree depth until an action centroid is reached.
                - With a probability 1-epsilon an action is sampled uniformly from
                  the centroid action space and with a probability epsilon an action
                  is uniformly sampled from action space.

            Args:
                x: the observations, i.e., batched contexts.
                epsilon: probability of selecting a random action.

            Returns:
                actions: sampled actions from the tree using epsilon-greedy
                        and scaled to the environment action range.
                probabilities: the probability density value of the actions.

            """

            tree = Tree(
                network_builder=self.network_builder,
                tree_params=self.tree_params,
            )
            logits = tree(x)

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

            (
                key_exploration,
                key_exploitation,
                key_sampling_exploration,
            ) = jax.random.split(hk.next_rng_key(), num=3)

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

            return actions, probabilities

        forward = hk.transform(_forward)

        _params = forward.init(rng=self.rng_key, x=obs, epsilon=epsilon)
        _forward_fn = jax.jit(forward.apply)

        return _params, _forward_fn

    def _create_forward_single_depth_fns(
        self, obs: Observations
    ) -> Tuple[Dict[int, Wrapped], Dict[int, hk.Params]]:
        """Creates a dictionary of jitted forward functions, one per neural networks at each tree depth
        and initializes the parameters of these neural networks.

        Args:
            obs: the observations, i.e., batched contexts.

        Returns:
            _forward_single_depth_fns: a dictionary of jitted forward functions of neural networks
                                    with tree depth as key.
            _depth_params: a dictionary of neural network parameters with tree depth as key.
        """

        def create_single_depth_function(
            depth: int,
        ) -> Callable[[JaxObservations], Logits]:
            """Creates a neural network forward function for a given depth.

            Args:
                depth: depth at which the neural network will be used.

            Returns:
                _forward: the neural network forward function.
            """

            n_leafs = 2 ** (depth + 1)

            def _forward(x: JaxObservations) -> Logits:
                """Creates a neural network forward function for a predefined depth.

                Args:
                    x: the observations, i.e., batched contexts.

                Returns:
                    the neural network forward function at the predefined depth.
                """

                tree = Tree(
                    network_builder=self.network_builder,
                    tree_params=self.tree_params,
                )
                return tree.networks[depth](x).reshape(x.shape[0], n_leafs // 2, 2)

            return _forward

        transformed_layers = {
            i: hk.transform(create_single_depth_function(i))
            for i in range(self.tree_params.depth)
        }
        _forward_single_depth_fns = {
            i: jax.jit(func.apply) for i, func in transformed_layers.items()
        }

        _depth_params = {
            i: layer.init(x=obs, rng=self.rng_key)
            for i, layer in transformed_layers.items()
        }

        return _forward_single_depth_fns, _depth_params

    def _init_opt_states(self) -> Dict[int, optax.OptState]:
        """Initializes an optimizer state for each tree depth
        using the parameters of the depth specific neural networks.

        Returns:
            a dictionary initialized optax optimizer states with tree depth as a key.
        """

        _opt_states = {
            d: self.optimizer.init(params) for d, params in self._depth_params.items()
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
    ) -> JaxLoss:
        """Computes the loss function a given depth.

        Args:
            layer_params: a dictionary of neural network parameters with tree depth as key.
            obs: the observations, i.e., batched contexts.
            smooth_costs: the smooth costs.
            depth: the tree depth at which the loss will be calculated.
            mask_eq: a mask with zeros where smooth cost pairs are equal
            rng_key: JAX key generator.

        Returns:
            the sum of the batch losses.
        """

        logits = self._forward_single_depth_fns[depth](
            params=layer_params, x=obs, rng=rng_key
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

        Returns:
            rng_key: JAX key generator.
            new_layer_params: a dictionary of the updated network parameters with tree depth as key.
            new_opt_states: a dictionary of updated optimizer states with tree depth as a key.

        """

        # Scale actions from the environment action range to the tree action range.
        actions = (actions - self._action_min) / (self._action_max - self._action_min)

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
                layers_params[depth], obs, smooth_costs, depth, mask_eq, loss_key
            )
            updates, opt_state = self.optimizer.update(grads, opt_states[depth])
            new_layer_params[depth] = optax.apply_updates(layers_params[depth], updates)
            new_opt_states[depth] = opt_state

            # Update smooth cost for the next upper tree depth
            if depth > 0:
                # Get smooth cost from 1 step forward following the updated parameters
                logits = self._forward_single_depth_fns[depth](
                    params=new_layer_params[depth], x=obs, rng=layer_key
                )
                mask = logits < jnp.max(logits, axis=-1).reshape(
                    logits.shape[0:2] + (1,)
                )
                c = jnp.where(mask, -jnp.inf, smooth_costs)
                smooth_costs = jnp.max(c, axis=-1).reshape(
                    logits.shape[0], logits.shape[1] // 2, 2
                )

        return rng_key, new_layer_params, new_opt_states
