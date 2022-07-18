from typing import Optional

import chex
import numpy as np

import tensorflow as tf
from sklearn.datasets import fetch_openml

from catx.type_defs import (
    Actions,
    Costs,
    Observations,
)


class OpenMLEnvironment:
    def __init__(self, dataset_id: int, batch_size: int = 5) -> None:

        self.x, self.y = fetch_openml(
            data_id=dataset_id, as_frame=False, return_X_y=True
        )

        rows_with_nan_idx = np.argwhere(np.isnan(self.x))[:, 0]
        self.x = np.delete(self.x, rows_with_nan_idx, axis=0)
        self.y = np.delete(self.y, rows_with_nan_idx, axis=0)
        self.x = self._normalize_data(self.x)
        self.y = self._normalize_data(self.y)
        self._y_mean = np.mean(self.y)
        physical_devices = tf.config.list_physical_devices("GPU")

        # Allow gpu memory growth for tensorflow.
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except Exception:
            pass  # Invalid device or cannot modify virtual devices once initialized.

        self.dataset = tf.data.Dataset.from_tensor_slices((self.x, self.y))
        self.dataset = self.dataset.batch(batch_size)
        self.iterator = iter(self.dataset)

    def get_new_observations(self, key: chex.PRNGKey) -> Optional[Observations]:
        del key
        try:
            x, y = self.iterator.get_next()
            self.x = x.numpy()
            self.y = y.numpy()
            return self.x
        except tf.errors.OutOfRangeError:
            return None

    def get_costs(
        self, key: chex.PRNGKey, obs: Observations, actions: Actions
    ) -> Costs:
        del key
        del obs
        costs = np.abs(actions - self.y)

        return costs

    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        return (data - np.min(data, axis=0)) / (
            np.max(data, axis=0) - np.min(data, axis=0)
        )
