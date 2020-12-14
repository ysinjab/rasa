import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import logging
import os
from collections import defaultdict
from typing import (
    List,
    Text,
    Dict,
    Tuple,
    Union,
    Optional,
    Any,
)

import rasa.utils.io
from rasa.constants import CHECKPOINT_MODEL_NAME
from rasa.utils.tensorflow.model_data import RasaModelData, FeatureSignature
from rasa.utils.tensorflow.constants import (
    SEQUENCE,
    SENTENCE,
    TENSORBOARD_LOG_LEVEL,
    RANDOM_SEED,
    TENSORBOARD_LOG_DIR,
    CHECKPOINT_MODEL,
    EMBEDDING_DIMENSION,
    REGULARIZATION_CONSTANT,
    SIMILARITY_TYPE,
    WEIGHT_SPARSITY,
    NUM_TRANSFORMER_LAYERS,
    TRANSFORMER_SIZE,
    NUM_HEADS,
    UNIDIRECTIONAL_ENCODER,
    KEY_RELATIVE_ATTENTION,
    VALUE_RELATIVE_ATTENTION,
    MAX_RELATIVE_POSITION,
    NUM_NEG,
    LOSS_TYPE,
    MAX_POS_SIM,
    MAX_NEG_SIM,
    USE_MAX_NEG_SIM,
    NEGATIVE_MARGIN_SCALE,
    HIDDEN_LAYERS_SIZES,
    DROP_RATE,
    DENSE_DIMENSION,
    CONCAT_DIMENSION,
    DROP_RATE_ATTENTION,
    SCALE_LOSS,
)
from rasa.utils.tensorflow import layers
from rasa.utils.tensorflow.transformer import TransformerEncoder
from rasa.utils.tensorflow.temp_keras_modules import TmpKerasModel
from rasa.utils.tensorflow.data_generator import (
    RasaDataGenerator,
    IncreasingBatchSizeDataGenerator,
)

logger = logging.getLogger(__name__)


# noinspection PyMethodOverriding
class RasaModel(TmpKerasModel):
    """Completely override all public methods of keras Model.

    Cannot be used as tf.keras.Model
    """

    def __init__(
        self,
        random_seed: Optional[int] = None,
        tensorboard_log_dir: Optional[Text] = None,
        tensorboard_log_level: Optional[Text] = "epoch",
        checkpoint_model: Optional[bool] = False,
        **kwargs,
    ) -> None:
        """Initialize the RasaModel.

        Args:
            random_seed: set the random seed to get reproducible results
        """
        super().__init__(**kwargs)

        self.total_loss = tf.keras.metrics.Mean(name="t_loss")
        self.metrics_to_log = ["t_loss"]

        self._training = None  # training phase should be defined when building a graph

        self._predict_function = None

        self.random_seed = random_seed

        tf.random.set_seed(self.random_seed)
        np.random.seed(self.random_seed)

        self.tensorboard_log_dir = tensorboard_log_dir
        self.tensorboard_log_level = tensorboard_log_level

        self.train_summary_writer = None
        self.test_summary_writer = None
        self.model_summary_file = None
        self.tensorboard_log_on_epochs = True

        self.best_metrics_so_far = {}
        self.checkpoint_model = checkpoint_model
        self.best_model_file = None
        self.best_model_epoch = -1
        if self.checkpoint_model:
            model_checkpoint_dir = rasa.utils.io.create_temporary_directory()
            self.best_model_file = os.path.join(
                model_checkpoint_dir, f"{CHECKPOINT_MODEL_NAME}.tf_model"
            )

    def batch_loss(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> tf.Tensor:
        """Calculates the loss for the given batch.

        Args:
            batch_in: The batch.

        Returns:
            The loss of the given batch.
        """
        raise NotImplementedError

    def batch_predict(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> Dict[Text, tf.Tensor]:
        """Predicts the output of the given batch.

        Args:
            batch_in: The batch.

        Returns:
            The output to predict.
        """
        raise NotImplementedError

    def train_step(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> Dict[Text, float]:
        """Train on batch."""
        # calculate supervision and regularization losses separately
        with tf.GradientTape(persistent=True) as tape:
            prediction_loss = self.batch_loss(batch_in)
            regularization_loss = tf.math.add_n(self.losses)
            total_loss = prediction_loss + regularization_loss

        self.total_loss.update_state(total_loss)

        # calculate the gradients that come from supervision signal
        prediction_gradients = tape.gradient(prediction_loss, self.trainable_variables)
        # calculate the gradients that come from regularization
        regularization_gradients = tape.gradient(
            regularization_loss, self.trainable_variables
        )
        # delete gradient tape manually
        # since it was created with `persistent=True` option
        del tape

        gradients = []
        for pred_grad, reg_grad in zip(prediction_gradients, regularization_gradients):
            if pred_grad is not None and reg_grad is not None:
                # remove regularization gradient for variables
                # that don't have prediction gradient
                gradients.append(
                    pred_grad
                    + tf.where(pred_grad > 0, reg_grad, tf.zeros_like(reg_grad))
                )
            else:
                gradients.append(pred_grad)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return self._get_metric_results()

    def test_step(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> Dict[Text, float]:
        """Train on batch."""
        prediction_loss = self.batch_loss(batch_in)
        regularization_loss = tf.math.add_n(self.losses)
        total_loss = prediction_loss + regularization_loss
        self.total_loss.update_state(total_loss)

        return self._get_metric_results()

    def predict_step(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> Dict[Text, tf.Tensor]:
        """Predict on batch."""
        self.prepare_for_predict()
        return self.batch_predict(batch_in)

    def _get_metric_results(self, prefix: Optional[Text] = None) -> Dict[Text, float]:
        """Get the metrics results."""
        prefix = prefix or ""

        return {
            f"{prefix}{metric.name}": metric.result()
            for metric in self.metrics
            if metric.name in self.metrics_to_log
        }

    def save(self, model_file_name: Text, overwrite: bool = True) -> None:
        """Save the model to the given file.

        Args:
            model_file_name: The file name to save the model to.
            overwrite: If 'True' an already existing model with the same file name will
                       be overwritten.
        """
        self.save_weights(model_file_name, overwrite=overwrite, save_format="tf")

    @classmethod
    def load(
        cls, model_file_name: Text, model_data_example: RasaModelData, *args, **kwargs
    ) -> "RasaModel":
        """Loads the model from the given file.

        Args:
            model_file_name: The model file.

        Returns:
            The loaded Rasa model.
        """
        logger.debug("Loading the model ...")
        # create empty model
        model = cls(*args, **kwargs)
        # need to train on 1 example to build weights of the correct size
        model.compile(run_eagerly=True)
        data_generator = IncreasingBatchSizeDataGenerator(
            model_data_example, batch_size=1
        )
        model.fit(data_generator, verbose=False)
        # load trained weights
        model.load_weights(model_file_name)

        logger.debug("Finished loading the model.")
        return model

    @staticmethod
    def batch_to_model_data_format(
        batch: Union[Tuple[tf.Tensor], Tuple[np.ndarray]],
        data_signature: Dict[Text, Dict[Text, List[FeatureSignature]]],
    ) -> Dict[Text, Dict[Text, List[tf.Tensor]]]:
        """Convert input batch tensors into batch data format.

        Batch contains any number of batch data. The order is equal to the
        key-value pairs in session data. As sparse data were converted into indices,
        data, shape before, this methods converts them into sparse tensors. Dense data
        is kept.
        """
        if isinstance(batch[0], Tuple):
            batch = batch[0]

        batch_data = defaultdict(lambda: defaultdict(list))

        idx = 0
        for key, values in data_signature.items():
            for sub_key, signature in values.items():
                for is_sparse, feature_dimension, number_of_dimensions in signature:
                    number_of_dimensions = (
                        number_of_dimensions if number_of_dimensions != 4 else 3
                    )
                    if is_sparse:
                        # explicitly substitute last dimension in shape with known
                        # static value
                        shape = [
                            batch[idx + 2][i] for i in range(number_of_dimensions - 1)
                        ] + [feature_dimension]
                        batch_data[key][sub_key].append(
                            tf.SparseTensor(batch[idx], batch[idx + 1], shape)
                        )
                        idx += 3
                    else:
                        if isinstance(batch[idx], tf.Tensor):
                            if number_of_dimensions > 1 and (
                                batch[idx].shape is None or batch[idx].shape[-1] is None
                            ):
                                shape = [None] * (number_of_dimensions - 1) + [
                                    feature_dimension
                                ]
                                batch[idx].set_shape(shape)
                            batch_data[key][sub_key].append(batch[idx])
                        else:
                            # convert to Tensor
                            batch_data[key][sub_key].append(
                                tf.constant(
                                    batch[idx], dtype=tf.float32, shape=batch[idx].shape
                                )
                            )
                        idx += 1

        return batch_data

    def call(
        self,
        inputs: Union[tf.Tensor, List[tf.Tensor]],
        training: Optional[tf.Tensor] = None,
        mask: Optional[tf.Tensor] = None,
    ) -> Union[tf.Tensor, List[tf.Tensor]]:
        """Calls the model on new inputs.

        Arguments:
            inputs: A tensor or list of tensors.
            training: Boolean or boolean scalar tensor, indicating whether to run
              the `Network` in training mode or inference mode.
            mask: A mask or list of masks. A mask can be
                either a tensor or None (no mask).

        Returns:
            A tensor if there is a single output, or
            a list of tensors if there are more than one outputs.
        """
        # This method needs to be implemented, otherwise the super class is raising a
        # NotImplementedError('When subclassing the `Model` class, you should
        #   implement a `call` method.')
        pass

    def prepare_for_predict(self) -> None:
        """Prepares tf graph for prediction.

        This method should contain necessary tf calculations
        and set self variables that are used in `batch_predict`.
        For example, pre calculation of `self.all_labels_embed`.
        """
        pass


# noinspection PyMethodOverriding
class TransformerRasaModel(RasaModel):
    def __init__(
        self,
        name: Text,
        config: Dict[Text, Any],
        data_signature: Dict[Text, Dict[Text, List[FeatureSignature]]],
        label_data: RasaModelData,
    ) -> None:
        super().__init__(
            name=name,
            random_seed=config[RANDOM_SEED],
            tensorboard_log_dir=config[TENSORBOARD_LOG_DIR],
            tensorboard_log_level=config[TENSORBOARD_LOG_LEVEL],
            checkpoint_model=config[CHECKPOINT_MODEL],
        )

        self.config = config
        self.data_signature = data_signature
        self.label_signature = label_data.get_signature()

        self._check_data()

        label_batch = RasaDataGenerator.prepare_batch(label_data.data)
        self.tf_label_data = self.batch_to_model_data_format(
            label_batch, self.label_signature
        )

        # set up tf layers
        self._tf_layers: Dict[Text, tf.keras.layers.Layer] = {}

    def _check_data(self) -> None:
        raise NotImplementedError

    def _prepare_layers(self) -> None:
        raise NotImplementedError

    def _prepare_embed_layers(self, name: Text, prefix: Text = "embed") -> None:
        self._tf_layers[f"{prefix}.{name}"] = layers.Embed(
            self.config[EMBEDDING_DIMENSION],
            self.config[REGULARIZATION_CONSTANT],
            name,
            self.config[SIMILARITY_TYPE],
        )

    def _prepare_ffnn_layer(
        self,
        name: Text,
        layer_sizes: List[int],
        drop_rate: float,
        prefix: Text = "ffnn",
    ) -> None:
        self._tf_layers[f"{prefix}.{name}"] = layers.Ffnn(
            layer_sizes,
            drop_rate,
            self.config[REGULARIZATION_CONSTANT],
            self.config[WEIGHT_SPARSITY],
            layer_name_suffix=name,
        )

    def _prepare_transformer_layer(
        self,
        name: Text,
        num_layers: int,
        units: int,
        drop_rate: float,
        drop_rate_attention: float,
        prefix: Text = "transformer",
    ):
        if self.config[NUM_TRANSFORMER_LAYERS] > 0:
            self._tf_layers[f"{prefix}.{name}"] = TransformerEncoder(
                num_layers,
                units,
                self.config[NUM_HEADS],
                units * 4,
                self.config[REGULARIZATION_CONSTANT],
                dropout_rate=drop_rate,
                attention_dropout_rate=drop_rate_attention,
                sparsity=self.config[WEIGHT_SPARSITY],
                unidirectional=self.config[UNIDIRECTIONAL_ENCODER],
                use_key_relative_position=self.config[KEY_RELATIVE_ATTENTION],
                use_value_relative_position=self.config[VALUE_RELATIVE_ATTENTION],
                max_relative_position=self.config[MAX_RELATIVE_POSITION],
                name=f"{name}_encoder",
            )
        else:
            # create lambda so that it can be used later without the check
            self._tf_layers[f"{prefix}.{name}"] = lambda x, mask, training: x

    def _prepare_dot_product_loss(
        self, name: Text, scale_loss: bool, prefix: Text = "loss"
    ) -> None:
        self._tf_layers[f"{prefix}.{name}"] = layers.DotProductLoss(
            self.config[NUM_NEG],
            self.config[LOSS_TYPE],
            self.config[MAX_POS_SIM],
            self.config[MAX_NEG_SIM],
            self.config[USE_MAX_NEG_SIM],
            self.config[NEGATIVE_MARGIN_SCALE],
            scale_loss,
            # set to 1 to get deterministic behaviour
            parallel_iterations=1 if self.random_seed is not None else 1000,
        )

    def _prepare_sparse_dense_dropout_layers(
        self, name: Text, drop_rate: float
    ) -> None:
        self._tf_layers[f"sparse_input_dropout.{name}"] = layers.SparseDropout(
            rate=drop_rate
        )
        self._tf_layers[f"dense_input_dropout.{name}"] = tf.keras.layers.Dropout(
            rate=drop_rate
        )

    def _prepare_sparse_dense_layers(
        self, data_signature: List[FeatureSignature], name: Text, dense_dim: int
    ) -> None:
        sparse = False
        dense = False
        for is_sparse, _, _ in data_signature:
            if is_sparse:
                sparse = True
            else:
                dense = True

        if sparse:
            self._tf_layers[f"sparse_to_dense.{name}"] = layers.DenseForSparse(
                units=dense_dim,
                reg_lambda=self.config[REGULARIZATION_CONSTANT],
                name=name,
            )
            if not dense:
                # create dense labels for the input to use in negative sampling
                self._tf_layers[f"sparse_to_dense_ids.{name}"] = layers.DenseForSparse(
                    units=2,
                    use_bias=False,
                    trainable=False,
                    name=f"sparse_to_dense_ids.{name}",
                )

    def _prepare_input_layers(self, name: Text) -> None:
        self._prepare_ffnn_layer(
            name, self.config[HIDDEN_LAYERS_SIZES][name], self.config[DROP_RATE]
        )

        for feature_type in [SENTENCE, SEQUENCE]:
            if (
                name not in self.data_signature
                or feature_type not in self.data_signature[name]
            ):
                continue

            self._prepare_sparse_dense_dropout_layers(
                f"{name}_{feature_type}", self.config[DROP_RATE]
            )
            self._prepare_sparse_dense_layers(
                self.data_signature[name][feature_type],
                f"{name}_{feature_type}",
                self.config[DENSE_DIMENSION][name],
            )
            self._prepare_ffnn_layer(
                f"{name}_{feature_type}",
                [self.config[CONCAT_DIMENSION][name]],
                self.config[DROP_RATE],
                prefix="concat_layer",
            )

    def _prepare_sequence_layers(self, name: Text) -> None:
        self._prepare_input_layers(name)
        self._prepare_transformer_layer(
            name,
            self.config[NUM_TRANSFORMER_LAYERS],
            self.config[TRANSFORMER_SIZE],
            self.config[DROP_RATE],
            self.config[DROP_RATE_ATTENTION],
        )

    def _prepare_entity_recognition_layers(self) -> None:
        for tag_spec in self._entity_tag_specs:
            name = tag_spec.tag_name
            num_tags = tag_spec.num_tags
            self._tf_layers[f"embed.{name}.logits"] = layers.Embed(
                num_tags, self.config[REGULARIZATION_CONSTANT], f"logits.{name}"
            )
            self._tf_layers[f"crf.{name}"] = layers.CRF(
                num_tags, self.config[REGULARIZATION_CONSTANT], self.config[SCALE_LOSS]
            )
            self._tf_layers[f"embed.{name}.tags"] = layers.Embed(
                self.config[EMBEDDING_DIMENSION],
                self.config[REGULARIZATION_CONSTANT],
                f"tags.{name}",
            )

    def _combine_sparse_dense_features(
        self,
        features: List[Union[np.ndarray, tf.Tensor, tf.SparseTensor]],
        name: Text,
        mask: Optional[tf.Tensor] = None,
        sparse_dropout: bool = False,
        dense_dropout: bool = False,
    ) -> Optional[tf.Tensor]:
        if not features:
            return None

        dense_features = []

        for f in features:
            if isinstance(f, tf.SparseTensor):
                if sparse_dropout:
                    _f = self._tf_layers[f"sparse_input_dropout.{name}"](
                        f, self._training
                    )
                else:
                    _f = f

                dense_f = self._tf_layers[f"sparse_to_dense.{name}"](_f)

                if dense_dropout:
                    dense_f = self._tf_layers[f"dense_input_dropout.{name}"](
                        dense_f, self._training
                    )

                dense_features.append(dense_f)
            else:
                dense_features.append(f)

        if mask is None:
            return tf.concat(dense_features, axis=-1)

        return tf.concat(dense_features, axis=-1) * mask

    def _combine_sequence_sentence_features(
        self,
        sequence_features: List[Union[tf.Tensor, tf.SparseTensor]],
        sentence_features: List[Union[tf.Tensor, tf.SparseTensor]],
        mask_sequence: tf.Tensor,
        mask_text: tf.Tensor,
        name: Text,
        sparse_dropout: bool = False,
        dense_dropout: bool = False,
    ) -> tf.Tensor:
        sequence_x = self._combine_sparse_dense_features(
            sequence_features,
            f"{name}_{SEQUENCE}",
            mask_sequence,
            sparse_dropout,
            dense_dropout,
        )
        sentence_x = self._combine_sparse_dense_features(
            sentence_features, f"{name}_{SENTENCE}", None, sparse_dropout, dense_dropout
        )

        if sequence_x is not None and sentence_x is None:
            return sequence_x

        if sequence_x is None and sentence_x is not None:
            return sentence_x

        if sequence_x is not None and sentence_x is not None:
            return self._concat_sequence_sentence_features(
                sequence_x, sentence_x, name, mask_text
            )

        raise ValueError(
            "No features are present. Please check your configuration file."
        )

    def _concat_sequence_sentence_features(
        self,
        sequence_x: tf.Tensor,
        sentence_x: tf.Tensor,
        name: Text,
        mask_text: tf.Tensor,
    ):
        if sequence_x.shape[-1] != sentence_x.shape[-1]:
            sequence_x = self._tf_layers[f"concat_layer.{name}_{SEQUENCE}"](
                sequence_x, self._training
            )
            sentence_x = self._tf_layers[f"concat_layer.{name}_{SENTENCE}"](
                sentence_x, self._training
            )

        # we need to concatenate the sequence features with the sentence features
        # we cannot use tf.concat as the sequence features are padded

        # (1) get position of sentence features in mask
        last = mask_text * tf.math.cumprod(
            1 - mask_text, axis=1, exclusive=True, reverse=True
        )
        # (2) multiply by sentence features so that we get a matrix of
        #     batch-dim x seq-dim x feature-dim with zeros everywhere except for
        #     for the sentence features
        sentence_x = last * sentence_x

        # (3) add a zero to the end of sequence matrix to match the final shape
        sequence_x = tf.pad(sequence_x, [[0, 0], [0, 1], [0, 0]])

        # (4) sum up sequence features and sentence features
        return sequence_x + sentence_x

    def _features_as_seq_ids(
        self, features: List[Union[np.ndarray, tf.Tensor, tf.SparseTensor]], name: Text
    ) -> Optional[tf.Tensor]:
        """Creates dense labels for negative sampling."""
        # if there are dense features - we can use them
        for f in features:
            if not isinstance(f, tf.SparseTensor):
                seq_ids = tf.stop_gradient(f)
                # add a zero to the seq dimension for the sentence features
                seq_ids = tf.pad(seq_ids, [[0, 0], [0, 1], [0, 0]])
                return seq_ids

        # use additional sparse to dense layer
        for f in features:
            if isinstance(f, tf.SparseTensor):
                seq_ids = tf.stop_gradient(
                    self._tf_layers[f"sparse_to_dense_ids.{name}"](f)
                )
                # add a zero to the seq dimension for the sentence features
                seq_ids = tf.pad(seq_ids, [[0, 0], [0, 1], [0, 0]])
                return seq_ids

        return None

    def _create_sequence(
        self,
        sequence_features: List[Union[tf.Tensor, tf.SparseTensor]],
        sentence_features: List[Union[tf.Tensor, tf.SparseTensor]],
        mask_sequence: tf.Tensor,
        mask: tf.Tensor,
        name: Text,
        sparse_dropout: bool = False,
        dense_dropout: bool = False,
        masked_lm_loss: bool = False,
        sequence_ids: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor, Optional[tf.Tensor], Optional[tf.Tensor]]:
        if sequence_ids:
            seq_ids = self._features_as_seq_ids(sequence_features, f"{name}_{SEQUENCE}")
        else:
            seq_ids = None

        inputs = self._combine_sequence_sentence_features(
            sequence_features,
            sentence_features,
            mask_sequence,
            mask,
            name,
            sparse_dropout,
            dense_dropout,
        )
        inputs = self._tf_layers[f"ffnn.{name}"](inputs, self._training)

        if masked_lm_loss:
            transformer_inputs, lm_mask_bool = self._tf_layers[f"{name}_input_mask"](
                inputs, mask, self._training
            )
        else:
            transformer_inputs = inputs
            lm_mask_bool = None

        outputs = self._tf_layers[f"transformer.{name}"](
            transformer_inputs, 1 - mask, self._training
        )

        if self.config[NUM_TRANSFORMER_LAYERS] > 0:
            # apply activation
            outputs = tfa.activations.gelu(outputs)

        return outputs, inputs, seq_ids, lm_mask_bool

    @staticmethod
    def _compute_mask(sequence_lengths: tf.Tensor) -> tf.Tensor:
        mask = tf.sequence_mask(sequence_lengths, dtype=tf.float32)
        # explicitly add last dimension to mask
        # to track correctly dynamic sequences
        return tf.expand_dims(mask, -1)

    @staticmethod
    def _last_token(x: tf.Tensor, sequence_lengths: tf.Tensor) -> tf.Tensor:
        last_sequence_index = tf.maximum(0, sequence_lengths - 1)
        batch_index = tf.range(tf.shape(last_sequence_index)[0])

        indices = tf.stack([batch_index, last_sequence_index], axis=1)
        return tf.gather_nd(x, indices)

    def _get_mask_for(
        self,
        tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]],
        key: Text,
        sub_key: Text,
    ) -> Optional[tf.Tensor]:
        if key not in tf_batch_data or sub_key not in tf_batch_data[key]:
            return None

        sequence_lengths = tf.cast(tf_batch_data[key][sub_key][0], dtype=tf.int32)
        return self._compute_mask(sequence_lengths)

    @staticmethod
    def _get_sequence_lengths(
        tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]],
        key: Text,
        sub_key: Text,
        batch_dim: int = 1,
    ) -> tf.Tensor:
        # sentence features have a sequence lengths of 1
        # if sequence features are present we add the sequence lengths of those

        sequence_lengths = tf.ones([batch_dim], dtype=tf.int32)
        if key in tf_batch_data and sub_key in tf_batch_data[key]:
            sequence_lengths += tf.cast(tf_batch_data[key][sub_key][0], dtype=tf.int32)

        return tf.cast(tf_batch_data[key][sub_key][0], dtype=tf.int32) + 1

    @staticmethod
    def _get_batch_dim(attribute_data: Dict[Text, List[tf.Tensor]]) -> int:
        if SEQUENCE in attribute_data:
            return tf.shape(attribute_data[SEQUENCE][0])[0]

        return tf.shape(attribute_data[SENTENCE][0])[0]

    def _calculate_entity_loss(
        self,
        inputs: tf.Tensor,
        tag_ids: tf.Tensor,
        mask: tf.Tensor,
        sequence_lengths: tf.Tensor,
        tag_name: Text,
        entity_tags: Optional[tf.Tensor] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

        tag_ids = tf.cast(tag_ids[:, :, 0], tf.int32)

        if entity_tags is not None:
            _tags = self._tf_layers[f"embed.{tag_name}.tags"](entity_tags)
            inputs = tf.concat([inputs, _tags], axis=-1)

        logits = self._tf_layers[f"embed.{tag_name}.logits"](inputs)

        # should call first to build weights
        pred_ids, _ = self._tf_layers[f"crf.{tag_name}"](logits, sequence_lengths)
        loss = self._tf_layers[f"crf.{tag_name}"].loss(
            logits, tag_ids, sequence_lengths
        )
        f1 = self._tf_layers[f"crf.{tag_name}"].f1_score(tag_ids, pred_ids, mask)

        return loss, f1, logits

    def batch_loss(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> tf.Tensor:
        """Calculates the loss for the given batch.

        Args:
            batch_in: The batch.

        Returns:
            The loss of the given batch.
        """
        raise NotImplementedError

    def batch_predict(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> Dict[Text, tf.Tensor]:
        """Predicts the output of the given batch.

        Args:
            batch_in: The batch.

        Returns:
            The output to predict.
        """
        raise NotImplementedError
