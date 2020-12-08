from typing import Text

import pytest

import scipy.sparse
import numpy as np

from rasa.utils.tensorflow.model_data import FeatureArray, RasaModelData
from rasa.utils.tensorflow.data_generator import (
    RasaDataGenerator,
    IncreasingBatchSizeDataGenerator,
    FixBatchSizeDataGenerator,
    FileLoadingDataGenerator,
)


def test_increasing_batch_size_generator(model_data: RasaModelData):
    epochs = 2

    data_generator = IncreasingBatchSizeDataGenerator(
        model_data,
        batch_size=[1, 2],
        epochs=epochs,
        batch_strategy="balanced",
        shuffle=True,
    )

    expected_batch_sizes = [[1, 1, 1, 1, 1], [2, 2, 1]]

    for _epoch in range(epochs):
        iterator = iter(data_generator)

        assert len(data_generator) == len(expected_batch_sizes[_epoch])

        for i in range(len(data_generator)):
            batch, _ = next(iterator)

            assert len(batch) == 11
            assert len(batch[0]) == expected_batch_sizes[_epoch][i]

        with pytest.raises(StopIteration):
            next(iterator)

        data_generator.on_epoch_end()


def test_fixed_batch_size_generator(model_data: RasaModelData):
    data_generator = FixBatchSizeDataGenerator(
        model_data, batch_size=2, epochs=1, batch_strategy="balanced", shuffle=True
    )

    expected_batch_sizes = [2, 2, 1]

    iterator = iter(data_generator)

    assert len(data_generator) == len(expected_batch_sizes)

    for i in range(len(data_generator)):
        batch, _ = next(iterator)
        assert len(batch) == 11
        assert len(batch[0]) == expected_batch_sizes[i]

    with pytest.raises(StopIteration):
        next(iterator)


def test_file_loading_data_generator(model_data: RasaModelData):
    data_chunks = [
        {"number_of_examples": 5, "file_path": "chunk1.tfrecord"},
        {"number_of_examples": 2, "file_path": "chunk2.tfrecord"},
        {"number_of_examples": 4, "file_path": "chunk3.tfrecord"},
        {"number_of_examples": 3, "file_path": "chunk4.tfrecord"},
        {"number_of_examples": 4, "file_path": "chunk5.tfrecord"},
    ]

    data_generator = FileLoadingDataGenerator(
        data_chunks, lambda x: RasaModelData(), batch_size=2
    )

    iterator = iter(data_generator)

    assert len(data_generator) == 10

    for i in range(len(data_generator)):
        batch, _ = next(iterator)

    with pytest.raises(StopIteration):
        next(iterator)


@pytest.mark.parametrize(
    "start, end, expected_file_path, expected_examples_processed_so_far",
    [
        (4, 6, "chunk1.tfrecord", 0),
        (0, 2, "chunk1.tfrecord", 0),
        (6, 8, "chunk2.tfrecord", 5),
    ],
)
def test_file_path_to_load(
    start: int,
    end: int,
    expected_file_path: Text,
    expected_examples_processed_so_far: int,
):
    data_chunks = [
        {"number_of_examples": 5, "file_path": "chunk1.tfrecord"},
        {"number_of_examples": 2, "file_path": "chunk2.tfrecord"},
    ]

    data_generator = FileLoadingDataGenerator(
        data_chunks, lambda x: RasaModelData(), batch_size=2, shuffle=False
    )

    file_path, examples_processed_so_far = data_generator._file_path_to_load(start, end)

    assert file_path == expected_file_path
    assert examples_processed_so_far == expected_examples_processed_so_far


@pytest.mark.parametrize(
    "incoming_data, expected_shape",
    [
        (FeatureArray(np.random.rand(7, 12), number_of_dimensions=2), (7, 12)),
        (FeatureArray(np.random.rand(7), number_of_dimensions=1), (7,)),
        (
            FeatureArray(
                np.array(
                    [
                        np.random.rand(1, 10),
                        np.random.rand(3, 10),
                        np.random.rand(7, 10),
                        np.random.rand(1, 10),
                    ]
                ),
                number_of_dimensions=3,
            ),
            (4, 7, 10),
        ),
        (
            FeatureArray(
                np.array(
                    [
                        np.array(
                            [
                                np.random.rand(1, 10),
                                np.random.rand(5, 10),
                                np.random.rand(7, 10),
                            ]
                        ),
                        np.array(
                            [
                                np.random.rand(1, 10),
                                np.random.rand(3, 10),
                                np.random.rand(3, 10),
                                np.random.rand(7, 10),
                            ]
                        ),
                        np.array([np.random.rand(2, 10)]),
                    ]
                ),
                number_of_dimensions=4,
            ),
            (8, 7, 10),
        ),
    ],
)
def test_pad_dense_data(incoming_data: FeatureArray, expected_shape: np.ndarray):
    padded_data = RasaDataGenerator._pad_dense_data(incoming_data)

    assert padded_data.shape == expected_shape


@pytest.mark.parametrize(
    "incoming_data, expected_shape",
    [
        (
            FeatureArray(
                np.array([scipy.sparse.csr_matrix(np.random.randint(5, size=(7, 12)))]),
                number_of_dimensions=3,
            ),
            [1, 7, 12],
        ),
        (
            FeatureArray(
                np.array([scipy.sparse.csr_matrix(np.random.randint(5, size=(7,)))]),
                number_of_dimensions=2,
            ),
            [1, 1, 7],
        ),
        (
            FeatureArray(
                np.array(
                    [
                        scipy.sparse.csr_matrix(np.random.randint(10, size=(1, 10))),
                        scipy.sparse.csr_matrix(np.random.randint(10, size=(3, 10))),
                        scipy.sparse.csr_matrix(np.random.randint(10, size=(7, 10))),
                        scipy.sparse.csr_matrix(np.random.randint(10, size=(1, 10))),
                    ]
                ),
                number_of_dimensions=3,
            ),
            (4, 7, 10),
        ),
        (
            FeatureArray(
                np.array(
                    [
                        np.array(
                            [
                                scipy.sparse.csr_matrix(
                                    np.random.randint(10, size=(1, 10))
                                ),
                                scipy.sparse.csr_matrix(
                                    np.random.randint(10, size=(5, 10))
                                ),
                                scipy.sparse.csr_matrix(
                                    np.random.randint(10, size=(7, 10))
                                ),
                            ]
                        ),
                        np.array(
                            [
                                scipy.sparse.csr_matrix(
                                    np.random.randint(10, size=(1, 10))
                                ),
                                scipy.sparse.csr_matrix(
                                    np.random.randint(10, size=(3, 10))
                                ),
                                scipy.sparse.csr_matrix(
                                    np.random.randint(10, size=(1, 10))
                                ),
                                scipy.sparse.csr_matrix(
                                    np.random.randint(10, size=(7, 10))
                                ),
                            ]
                        ),
                        np.array(
                            [
                                scipy.sparse.csr_matrix(
                                    np.random.randint(10, size=(2, 10))
                                )
                            ]
                        ),
                    ]
                ),
                number_of_dimensions=4,
            ),
            (8, 7, 10),
        ),
    ],
)
def test_scipy_matrix_to_values(
    incoming_data: FeatureArray, expected_shape: np.ndarray
):
    indices, data, shape = RasaDataGenerator._scipy_matrix_to_values(incoming_data)

    assert np.all(shape == expected_shape)
