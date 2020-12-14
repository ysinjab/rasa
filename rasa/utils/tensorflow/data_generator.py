from typing import List, Union, Text, Optional, Any, Tuple, Dict

import scipy.sparse
import numpy as np
import tensorflow as tf

import rasa.shared.utils.io
from rasa.utils.tensorflow.constants import SEQUENCE, BALANCED
from rasa.utils.tensorflow.model_data import RasaModelData, Data, FeatureArray


class RasaDataGenerator(tf.keras.utils.Sequence):
    """Abstract data generator."""

    def __init__(
        self,
        model_data: RasaModelData,
        batch_size: Union[int, List[int]],
        epochs: int,
        batch_strategy: Text = SEQUENCE,
        shuffle: bool = True,
    ):
        """Initializes the data generator.

        Args:
            model_data: The model data to use.
            batch_size: The batch size(s).
            epochs: The total number of epochs.
            batch_strategy: The batch strategy.
            shuffle: If 'True', data should be shuffled.
        """
        self.model_data = model_data
        self.batch_size = batch_size
        self.epochs = epochs
        self.shuffle = shuffle
        self.batch_strategy = batch_strategy

        self.on_epoch_end()

    def __len__(self) -> int:
        """Number of batches in the Sequence.

        Returns:
            The number of batches in the Sequence.
        """
        pass

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Gets batch at position `index`.

        Arguments:
            index: position of the batch in the Sequence.

        Returns:
            A batch (tuple of input data and target data).
        """
        pass

    def on_epoch_end(self) -> None:
        """Update the data after every epoch."""
        pass

    def _shuffle_and_balance(self, batch_size: int):
        data = self.model_data.data

        if self.shuffle:
            data = self.model_data.shuffled_data(data)

        if self.batch_strategy == BALANCED:
            data = self.model_data.balanced_data(data, batch_size, self.shuffle)

        self.model_data.data = data

    @staticmethod
    def prepare_batch(
        data: Data,
        start: Optional[int] = None,
        end: Optional[int] = None,
        tuple_sizes: Optional[Dict[Text, int]] = None,
    ) -> Tuple[Optional[np.ndarray]]:
        """Slices model data into batch using given start and end value.

        Args:
            data: The data to prepare.
            start: The start index of the batch
            end: The end index of the batch
            tuple_sizes: In case the feature is not present we propagate the batch with
              None. Tuple sizes contains the number of how many None values to add for
              what kind of feature.

        Returns:
            The features of the batch.
        """
        batch_data = []

        for key, attribute_data in data.items():
            for sub_key, f_data in attribute_data.items():
                # add None for not present values during processing
                if not f_data:
                    if tuple_sizes:
                        batch_data += [None] * tuple_sizes[key]
                    else:
                        batch_data.append(None)
                    continue

                for v in f_data:
                    if start is not None and end is not None:
                        _data = v[start:end]
                    elif start is not None:
                        _data = v[start:]
                    elif end is not None:
                        _data = v[:end]
                    else:
                        _data = v[:]

                    if _data.is_sparse:
                        batch_data.extend(
                            RasaDataGenerator._scipy_matrix_to_values(_data)
                        )
                    else:
                        batch_data.append(RasaDataGenerator._pad_dense_data(_data))

        # len of batch_data is equal to the number of keys in model data
        return tuple(batch_data)

    @staticmethod
    def _pad_dense_data(array_of_dense: FeatureArray) -> np.ndarray:
        """Pad data of different lengths.

        Sequential data is padded with zeros. Zeros are added to the end of data.

        Args:
            array_of_dense: The array to pad.

        Returns:
            The padded array.
        """
        if array_of_dense.number_of_dimensions == 4:
            return RasaDataGenerator._pad_4d_dense_data(array_of_dense)

        if array_of_dense[0].ndim < 2:
            # data doesn't contain a sequence
            return array_of_dense.astype(np.float32)

        data_size = len(array_of_dense)
        max_seq_len = max([x.shape[0] for x in array_of_dense])

        data_padded = np.zeros(
            [data_size, max_seq_len, array_of_dense[0].shape[-1]],
            dtype=array_of_dense[0].dtype,
        )
        for i in range(data_size):
            data_padded[i, : array_of_dense[i].shape[0], :] = array_of_dense[i]

        return data_padded.astype(np.float32)

    @staticmethod
    def _pad_4d_dense_data(array_of_array_of_dense: FeatureArray) -> np.ndarray:
        # in case of dialogue data we may have 4 dimensions
        # batch size x dialogue history length x sequence length x number of features

        # as transformers cannot handle 4D tensors pad and reshape the data
        # so that the resulting tensor is 3D
        # the shape is (sum of dialogue history length for all tensors in the
        # batch x max sequence length x number of features)
        # the original shape and the original dialogue length is passed on to the model
        # it can be used to transform the 3D tensor back into 4D

        # in order to create 4d tensor inputs, we created "fake" zero features
        # for nonexistent inputs. To save calculation we filter this features before
        # input to tf methods.
        number_of_features = array_of_array_of_dense[0][0].shape[-1]
        array_of_array_of_dense = RasaDataGenerator._filter_out_fake_inputs(
            array_of_array_of_dense
        )
        if not array_of_array_of_dense:
            # return empty 3d array with appropriate last dims
            return np.zeros((0, 0, number_of_features), dtype=np.float32)

        combined_dialogue_len = sum(
            len(array_of_dense) for array_of_dense in array_of_array_of_dense
        )
        max_seq_len = max(
            [
                x.shape[0]
                for array_of_dense in array_of_array_of_dense
                for x in array_of_dense
            ]
        )

        data_padded = np.zeros(
            [combined_dialogue_len, max_seq_len, number_of_features],
            dtype=array_of_array_of_dense[0][0].dtype,
        )

        current_sum_dialogue_len = 0
        for i, array_of_dense in enumerate(array_of_array_of_dense):
            for j, dense in enumerate(array_of_dense):
                data_padded[current_sum_dialogue_len + j, : dense.shape[0], :] = dense
            current_sum_dialogue_len += len(array_of_dense)

        return data_padded.astype(np.float32)

    @staticmethod
    def _scipy_matrix_to_values(array_of_sparse: FeatureArray) -> List[np.ndarray]:
        """Convert a scipy matrix into indices, data, and shape.

        Args:
            array_of_sparse: The sparse data array.

        Returns:
            A list of dense numpy arrays representing the sparse data.
        """
        if array_of_sparse.number_of_dimensions == 4:
            return RasaDataGenerator._4d_scipy_matrix_to_values(array_of_sparse)

        # we need to make sure that the matrices are coo_matrices otherwise the
        # transformation does not work (e.g. you cannot access x.row, x.col)
        if not isinstance(array_of_sparse[0], scipy.sparse.coo_matrix):
            array_of_sparse = [x.tocoo() for x in array_of_sparse]

        max_seq_len = max([x.shape[0] for x in array_of_sparse])

        # get the indices of values
        indices = np.hstack(
            [
                np.vstack([i * np.ones_like(x.row), x.row, x.col])
                for i, x in enumerate(array_of_sparse)
            ]
        ).T

        data = np.hstack([x.data for x in array_of_sparse])

        number_of_features = array_of_sparse[0].shape[-1]
        shape = np.array((len(array_of_sparse), max_seq_len, number_of_features))

        return [
            indices.astype(np.int64),
            data.astype(np.float32),
            shape.astype(np.int64),
        ]

    @staticmethod
    def _4d_scipy_matrix_to_values(
        array_of_array_of_sparse: FeatureArray,
    ) -> List[np.ndarray]:
        # in case of dialogue data we may have 4 dimensions
        # batch size x dialogue history length x sequence length x number of features

        # transformers cannot handle 4D tensors, therefore pad and reshape the data
        # so that the resulting tensor is 3D
        # the shape is (sum of dialogue history length for all tensors in the
        # batch x max sequence length x number of features)
        # the original shape and the original dialogue length is passed on to the model
        # it can be used to transform the 3D tensor back into 4D

        # in order to create 4d tensor inputs, we created "fake" zero features
        # for nonexistent inputs. To save calculation we filter this features before
        # input to tf methods.
        number_of_features = array_of_array_of_sparse[0][0].shape[-1]
        array_of_array_of_sparse = RasaDataGenerator._filter_out_fake_inputs(
            array_of_array_of_sparse
        )
        if not array_of_array_of_sparse:
            # create empty array with appropriate last dims
            return [
                np.empty((0, 3), dtype=np.int64),
                np.array([], dtype=np.float32),
                np.array([0, 0, number_of_features], dtype=np.int64),
            ]

        # we need to make sure that the matrices are coo_matrices otherwise the
        # transformation does not work (e.g. you cannot access x.row, x.col)
        if not isinstance(array_of_array_of_sparse[0][0], scipy.sparse.coo_matrix):
            array_of_array_of_sparse = [
                [x.tocoo() for x in array_of_sparse]
                for array_of_sparse in array_of_array_of_sparse
            ]

        dialogue_len = [
            len(array_of_sparse) for array_of_sparse in array_of_array_of_sparse
        ]
        combined_dialogue_len = sum(dialogue_len)
        max_seq_len = max(
            [
                x.shape[0]
                for array_of_sparse in array_of_array_of_sparse
                for x in array_of_sparse
            ]
        )
        # get the indices of values
        indices = np.hstack(
            [
                np.vstack(
                    [sum(dialogue_len[:i]) + j * np.ones_like(x.row), x.row, x.col]
                )
                for i, array_of_sparse in enumerate(array_of_array_of_sparse)
                for j, x in enumerate(array_of_sparse)
            ]
        ).T

        data = np.hstack(
            [
                x.data
                for array_of_sparse in array_of_array_of_sparse
                for x in array_of_sparse
            ]
        )

        shape = np.array((combined_dialogue_len, max_seq_len, number_of_features))

        return [
            indices.astype(np.int64),
            data.astype(np.float32),
            shape.astype(np.int64),
        ]

    @staticmethod
    def _filter_out_fake_inputs(
        array_of_array_of_features: FeatureArray,
    ) -> Union[List[List[np.ndarray]], List[List[scipy.sparse.spmatrix]]]:
        return list(
            filter(
                # filter empty lists created by another filter
                lambda x: len(x) > 0,
                [
                    # filter all the "fake" inputs, we know the input is "fake",
                    # when sequence dimension is `0`
                    list(filter(lambda x: x.shape[0] > 0, array_of_features))
                    for array_of_features in array_of_array_of_features
                ],
            )
        )


class IncreasingBatchSizeDataGenerator(RasaDataGenerator):
    """Data generator with an optional increasing batch size."""

    def __init__(
        self,
        model_data: RasaModelData,
        batch_size: Union[List[int], int],
        epochs: int = 1,
        batch_strategy: Text = SEQUENCE,
        shuffle: bool = True,
    ):
        """Initializes the increasing batch size data generator.

        Args:
            model_data: The model data to use.
            batch_size: The batch size.
            epochs: The total number of epochs.
            batch_strategy: The batch strategy.
            shuffle: If 'True', data will be shuffled.
        """
        self.current_epoch = -1
        self.current_batch_size = None

        super().__init__(model_data, batch_size, epochs, batch_strategy, shuffle)

    def __len__(self) -> int:
        """Number of batches in the Sequence.

        Returns:
            The number of batches in the Sequence.
        """
        num_examples = self.model_data.num_examples
        batch_size = self.current_batch_size
        return num_examples // batch_size + int(num_examples % batch_size > 0)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Gets batch at position `index`.

        Arguments:
            index: position of the batch in the Sequence.

        Returns:
            A batch (tuple of input data and target data).
        """
        start = index * self.current_batch_size
        end = start + self.current_batch_size

        return self.prepare_batch(self.model_data.data, start, end), None

    def on_epoch_end(self) -> None:
        """Update the data after every epoch."""
        self.current_epoch += 1
        self.current_batch_size = self.linearly_increasing_batch_size(
            self.current_epoch, self.batch_size, self.epochs
        )
        self._shuffle_and_balance(self.current_batch_size)

    @staticmethod
    def linearly_increasing_batch_size(
        epoch: int, batch_size: Union[List[int], int], epochs: int
    ) -> int:
        """Linearly increase batch size with every epoch.

        The idea comes from https://arxiv.org/abs/1711.00489.

        Args:
            epoch: The current epoch number.
            batch_size: The batch sizes to use.
            epochs: The total number of epochs.

        Returns:
            The batch size to use in this epoch.
        """
        if not isinstance(batch_size, list):
            return int(batch_size)

        if epochs > 1:
            return int(
                batch_size[0] + epoch * (batch_size[1] - batch_size[0]) / (epochs - 1)
            )
        else:
            return int(batch_size[0])


class FixBatchSizeDataGenerator(RasaDataGenerator):
    """Data generator with a fixed batch size."""

    def __init__(
        self,
        model_data: RasaModelData,
        batch_size: int,
        epochs: int = 1,
        batch_strategy: Text = SEQUENCE,
        shuffle: bool = True,
    ):
        """Initializes the increasing batch size data generator.

        Args:
            model_data: The model data to use.
            batch_size: The batch size.
            epochs: The total number of epochs.
            batch_strategy: The batch strategy.
            shuffle: If 'True', data will be shuffled.
        """
        if not isinstance(batch_size, int):
            rasa.shared.utils.io.raise_warning(
                f"'FixBatchSizeDataGenerator' should only be used with a "
                f"fixed batch size, but '{batch_size}' given. Use default"
                f"batch size of 32 instead."
            )
            batch_size = 32

        super().__init__(model_data, batch_size, epochs, batch_strategy, shuffle)

    def __len__(self) -> int:
        """Number of batches in the Sequence.

        Returns:
            The number of batches in the Sequence.
        """
        num_examples = self.model_data.num_examples
        return num_examples // self.batch_size + int(num_examples % self.batch_size > 0)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Gets batch at position `index`.

        Arguments:
            index: position of the batch in the Sequence.

        Returns:
            A batch (tuple of input data and target data).
        """
        start = index * self.batch_size
        end = start + self.batch_size

        return self.prepare_batch(self.model_data.data, start, end), None

    def on_epoch_end(self) -> None:
        """Update the data after every epoch."""
        self._shuffle_and_balance(self.batch_size)
