from typing import List, Union, Text, Optional, Any, Iterator, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.gen_dataset_ops import iterator

from rasa.utils.tensorflow.constants import SEQUENCE, BALANCED
from rasa.utils.tensorflow.model_data import RasaModelData


class IncreasingBatchSizeDataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        model_data: RasaModelData,
        batch_size: List[int],
        epochs: int,
        batch_strategy: Text = SEQUENCE,
        shuffle: bool = True,
    ):
        self.model_data = model_data

        self.batch_size = batch_size
        self.current_batch_size = None

        self.current_epoch = -1
        self.epochs = epochs

        self.shuffle = shuffle
        self.batch_strategy = batch_strategy

        self.on_epoch_end()

    def __len__(self) -> int:
        num_examples = self.model_data.num_examples
        batch_size = self.current_batch_size
        len = num_examples // batch_size + int(num_examples % batch_size > 0)
        return len

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        return self._gen_batch(index), None, None

    def on_epoch_end(self) -> None:
        self.current_epoch += 1
        self.current_batch_size = self.linearly_increasing_batch_size(
            self.current_epoch, self.batch_size, self.epochs
        )
        self._shuffle_and_balance()

    def _shuffle_and_balance(self):
        data = self.model_data.data

        if self.shuffle:
            data = self.model_data.shuffled_data(data)

        if self.batch_strategy == BALANCED:
            data = self.model_data.balanced_data(
                data, self.current_batch_size, self.shuffle
            )

        self.model_data.data = data

    def _gen_batch(self, index: int) -> Tuple[Optional[np.ndarray]]:
        start = index * self.current_batch_size
        end = start + self.current_batch_size

        return self.model_data.prepare_batch(self.model_data.data, start, end)

    @staticmethod
    def linearly_increasing_batch_size(
        epoch: int, batch_size: Union[List[int], int], epochs: int
    ) -> int:
        """Linearly increase batch size with every epoch.

        The idea comes from https://arxiv.org/abs/1711.00489.
        """
        if not isinstance(batch_size, list):
            return int(batch_size)

        if epochs > 1:
            return int(
                batch_size[0] + epoch * (batch_size[1] - batch_size[0]) / (epochs - 1)
            )
        else:
            return int(batch_size[0])
