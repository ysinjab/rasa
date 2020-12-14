from pathlib import Path
from typing import Optional, Text, Dict, Any, Union, List, Tuple, TYPE_CHECKING

import tensorflow as tf
import numpy as np

import rasa.shared.utils.common
import rasa.shared.utils.io
import rasa.nlu.utils.bilou_utils
from rasa.shared.constants import NEXT_MAJOR_VERSION_FOR_DEPRECATIONS
from rasa.nlu.constants import NUMBER_OF_SUB_TOKENS
from rasa.nlu.tokenizers.tokenizer import Token
import rasa.utils.io as io_utils
from rasa.utils.tensorflow.constants import (
    LOSS_TYPE,
    SIMILARITY_TYPE,
    EVAL_NUM_EXAMPLES,
    EVAL_NUM_EPOCHS,
    EPOCHS,
    SOFTMAX,
    MARGIN,
    AUTO,
    INNER,
    COSINE,
    SEQUENCE,
)
from rasa.utils.tensorflow.callback import RasaTrainingLogger, RasaModelCheckpoint
from rasa.utils.tensorflow.data_generator import RasaBatchDataGenerator
from rasa.utils.tensorflow.model_data import RasaModelData

if TYPE_CHECKING:
    from rasa.nlu.classifiers.diet_classifier import EntityTagSpec


def normalize(values: np.ndarray, ranking_length: Optional[int] = 0) -> np.ndarray:
    """Normalizes an array of positive numbers over the top `ranking_length` values.

    Other values will be set to 0.
    """
    new_values = values.copy()  # prevent mutation of the input
    if 0 < ranking_length < len(new_values):
        ranked = sorted(new_values, reverse=True)
        new_values[new_values < ranked[ranking_length - 1]] = 0

    if np.sum(new_values) > 0:
        new_values = new_values / np.sum(new_values)

    return new_values


def update_similarity_type(config: Dict[Text, Any]) -> Dict[Text, Any]:
    """
    If SIMILARITY_TYPE is set to 'auto', update the SIMILARITY_TYPE depending
    on the LOSS_TYPE.
    Args:
        config: model configuration

    Returns: updated model configuration
    """
    if config.get(SIMILARITY_TYPE) == AUTO:
        if config[LOSS_TYPE] == SOFTMAX:
            config[SIMILARITY_TYPE] = INNER
        elif config[LOSS_TYPE] == MARGIN:
            config[SIMILARITY_TYPE] = COSINE

    return config


def align_token_features(
    list_of_tokens: List[List[Token]],
    in_token_features: np.ndarray,
    shape: Optional[Tuple] = None,
) -> np.ndarray:
    """Align token features to match tokens.

    ConveRTTokenizer, LanguageModelTokenizers might split up tokens into sub-tokens.
    We need to take the mean of the sub-token vectors and take that as token vector.

    Args:
        list_of_tokens: tokens for examples
        in_token_features: token features from ConveRT
        shape: shape of feature matrix

    Returns:
        Token features.
    """
    if shape is None:
        shape = in_token_features.shape
    out_token_features = np.zeros(shape)

    for example_idx, example_tokens in enumerate(list_of_tokens):
        offset = 0
        for token_idx, token in enumerate(example_tokens):
            number_sub_words = token.get(NUMBER_OF_SUB_TOKENS, 1)

            if number_sub_words > 1:
                token_start_idx = token_idx + offset
                token_end_idx = token_idx + offset + number_sub_words

                mean_vec = np.mean(
                    in_token_features[example_idx][token_start_idx:token_end_idx],
                    axis=0,
                )

                offset += number_sub_words - 1

                out_token_features[example_idx][token_idx] = mean_vec
            else:
                out_token_features[example_idx][token_idx] = in_token_features[
                    example_idx
                ][token_idx + offset]

    return out_token_features


def update_evaluation_parameters(config: Dict[Text, Any]) -> Dict[Text, Any]:
    """
    If EVAL_NUM_EPOCHS is set to -1, evaluate at the end of the training.

    Args:
        config: model configuration

    Returns: updated model configuration
    """

    if config[EVAL_NUM_EPOCHS] == -1:
        config[EVAL_NUM_EPOCHS] = config[EPOCHS]
    elif config[EVAL_NUM_EPOCHS] < 1:
        raise ValueError(
            f"'{EVAL_NUM_EXAMPLES}' is set to "
            f"'{config[EVAL_NUM_EPOCHS]}'. "
            f"Only values > 1 are allowed for this configuration value."
        )

    return config


def load_tf_hub_model(model_url: Text) -> Any:
    """Load model from cache if possible, otherwise from TFHub"""

    import tensorflow_hub as tfhub

    # needed to load the ConveRT model
    # noinspection PyUnresolvedReferences
    import tensorflow_text
    import os

    # required to take care of cases when other files are already
    # stored in the default TFHUB_CACHE_DIR
    try:
        return tfhub.load(model_url)
    except OSError:
        directory = io_utils.create_temporary_directory()
        os.environ["TFHUB_CACHE_DIR"] = directory
        return tfhub.load(model_url)


def _replace_deprecated_option(
    old_option: Text,
    new_option: Union[Text, List[Text]],
    config: Dict[Text, Any],
    warn_until_version: Text = NEXT_MAJOR_VERSION_FOR_DEPRECATIONS,
) -> Dict[Text, Any]:
    if old_option in config:
        if isinstance(new_option, str):
            rasa.shared.utils.io.raise_deprecation_warning(
                f"Option '{old_option}' got renamed to '{new_option}'. "
                f"Please update your configuration file.",
                warn_until_version=warn_until_version,
            )
            config[new_option] = config[old_option]
        else:
            rasa.shared.utils.io.raise_deprecation_warning(
                f"Option '{old_option}' got renamed to "
                f"a dictionary '{new_option[0]}' with a key '{new_option[1]}'. "
                f"Please update your configuration file.",
                warn_until_version=warn_until_version,
            )
            option_dict = config.get(new_option[0], {})
            option_dict[new_option[1]] = config[old_option]
            config[new_option[0]] = option_dict

    return config


def check_deprecated_options(config: Dict[Text, Any]) -> Dict[Text, Any]:
    """
    If old model configuration parameters are present in the provided config, replace
    them with the new parameters and log a warning.
    Args:
        config: model configuration

    Returns: updated model configuration
    """

    # note: call _replace_deprecated_option() here when there are options to deprecate

    return config


def create_data_generators(
    model_data: RasaModelData,
    batch_sizes: Union[int, List[int]],
    epochs: int,
    batch_strategy: Text = SEQUENCE,
    eval_num_examples: int = 0,
    random_seed: Optional[int] = None,
) -> Tuple[RasaBatchDataGenerator, Optional[RasaBatchDataGenerator]]:
    """Create data generators for train and optional validation data.

    Args:
        model_data: The model data to use.
        batch_sizes: The batch size(s).
        epochs: The number of epochs to train.
        batch_strategy: The batch strategy to use.
        eval_num_examples: Number of examples to use for validation data.
        random_seed: The random seed.

    Returns:
        The training data generator and optional validation data generator.
    """
    validation_data_generator = None
    if eval_num_examples > 0:
        model_data, evaluation_model_data = model_data.split(
            eval_num_examples, random_seed,
        )
        validation_data_generator = RasaBatchDataGenerator(
            evaluation_model_data,
            batch_size=batch_sizes,
            epochs=epochs,
            batch_strategy=batch_strategy,
            shuffle=True,
        )

    data_generator = RasaBatchDataGenerator(
        model_data,
        batch_size=batch_sizes,
        epochs=epochs,
        batch_strategy=batch_strategy,
        shuffle=True,
    )

    return data_generator, validation_data_generator


def create_common_callbacks(
    epochs: int,
    tensorboard_log_dir: Optional[Text] = None,
    tensorboard_log_level: Optional[Text] = None,
    checkpoint_dir: Optional[Path] = None,
) -> List[tf.keras.callbacks.Callback]:
    """Create common callbacks.

    The following callbacks are created:
    - RasaTrainingLogger callback
    - Optional TensorBoard callback
    - Optional RasaModelCheckpoint callback

    Args:
        epochs: the number of epochs to train
        tensorboard_log_dir: optional directory that should be used for tensorboard
        tensorboard_log_level: defines when training metrics for tensorboard should be
                               logged. Valid values: 'epoch' and 'batch'.
        checkpoint_dir: optional directory that should be used for model checkpointing

    Returns:
        A list of callbacks.
    """
    callbacks = [RasaTrainingLogger(epochs, silent=False)]

    if tensorboard_log_dir:
        if tensorboard_log_level == "minibatch":
            tensorboard_log_level = "batch"
            rasa.shared.utils.io.raise_warning(
                "You set 'tensorboard_log_level' to 'minibatch'. This value should not "
                "be used anymore. Please use 'batch' instead."
            )

        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=tensorboard_log_dir,
                update_freq=tensorboard_log_level,
                write_graph=True,
                write_images=True,
                histogram_freq=10,
            )
        )

    if checkpoint_dir:
        callbacks.append(RasaModelCheckpoint(checkpoint_dir))

    return callbacks


def entity_label_to_tags(
    model_predictions: Dict[Text, Any],
    entity_tag_specs: List["EntityTagSpec"],
    bilou_flag: bool = False,
    prediction_index: int = 0,
) -> Tuple[Dict[Text, List[Text]], Dict[Text, List[float]]]:
    """Convert the output predictions for entities to the actual entity tags.

    Args:
        model_predictions: the output predictions using the entity tag indices
        entity_tag_specs: the entity tag specifications
        bilou_flag: if 'True', the BILOU tagging schema was used
        prediction_index: the index in the batch of predictions
            to use for entity extraction

    Returns:
        A map of entity tag type, e.g. entity, role, group, to actual entity tags and
        confidences.
    """
    predicted_tags = {}
    confidence_values = {}

    for tag_spec in entity_tag_specs:
        predictions = model_predictions[f"e_{tag_spec.tag_name}_ids"]
        confidences = model_predictions[f"e_{tag_spec.tag_name}_scores"]

        if not np.any(predictions):
            continue

        confidences = [float(c) for c in confidences[prediction_index]]
        tags = [tag_spec.ids_to_tags[p] for p in predictions[prediction_index]]

        if bilou_flag:
            (
                tags,
                confidences,
            ) = rasa.nlu.utils.bilou_utils.ensure_consistent_bilou_tagging(
                tags, confidences
            )

        predicted_tags[tag_spec.tag_name] = tags
        confidence_values[tag_spec.tag_name] = confidences

    return predicted_tags, confidence_values
