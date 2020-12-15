from pathlib import Path
from typing import Text

import numpy as np
import pytest
from unittest.mock import Mock

from rasa.shared.nlu.training_data.features import Features
from rasa.nlu import train
from rasa.nlu.classifiers import LABEL_RANKING_LENGTH
from rasa.nlu.config import RasaNLUModelConfig
from rasa.shared.nlu.constants import (
    TEXT,
    INTENT,
    FEATURE_TYPE_SENTENCE,
    FEATURE_TYPE_SEQUENCE,
)
from rasa.utils.tensorflow.constants import (
    LOSS_TYPE,
    RANDOM_SEED,
    RANKING_LENGTH,
    EPOCHS,
    MASKED_LM,
    TENSORBOARD_LOG_LEVEL,
    TENSORBOARD_LOG_DIR,
    EVAL_NUM_EPOCHS,
    EVAL_NUM_EXAMPLES,
    CHECKPOINT_MODEL,
    BILOU_FLAG,
    ENTITY_RECOGNITION,
    INTENT_CLASSIFICATION,
)
from rasa.nlu.components import ComponentBuilder
from rasa.nlu.classifiers.diet_classifier import DIETClassifier
from rasa.nlu.model import Interpreter
from rasa.shared.nlu.training_data.message import Message
from rasa.utils import train_utils
from tests.conftest import DEFAULT_NLU_DATA
from tests.nlu.conftest import DEFAULT_DATA_PATH


def test_compute_default_label_features():
    label_features = [
        Message(data={TEXT: "test a"}),
        Message(data={TEXT: "test b"}),
        Message(data={TEXT: "test c"}),
        Message(data={TEXT: "test d"}),
    ]

    output = DIETClassifier._compute_default_label_features(label_features)

    output = output[0]

    for i, o in enumerate(output):
        assert isinstance(o, np.ndarray)
        assert o[0][i] == 1
        assert o.shape == (1, len(label_features))


@pytest.mark.parametrize(
    "messages, expected",
    [
        (
            [
                Message(
                    data={TEXT: "test a"},
                    features=[
                        Features(np.zeros(1), FEATURE_TYPE_SEQUENCE, TEXT, "test"),
                        Features(np.zeros(1), FEATURE_TYPE_SENTENCE, TEXT, "test"),
                    ],
                ),
                Message(
                    data={TEXT: "test b"},
                    features=[
                        Features(np.zeros(1), FEATURE_TYPE_SEQUENCE, TEXT, "test"),
                        Features(np.zeros(1), FEATURE_TYPE_SENTENCE, TEXT, "test"),
                    ],
                ),
            ],
            True,
        ),
        (
            [
                Message(
                    data={TEXT: "test a"},
                    features=[
                        Features(np.zeros(1), FEATURE_TYPE_SEQUENCE, INTENT, "test"),
                        Features(np.zeros(1), FEATURE_TYPE_SENTENCE, INTENT, "test"),
                    ],
                )
            ],
            False,
        ),
        (
            [
                Message(
                    data={TEXT: "test a"},
                    features=[
                        Features(np.zeros(2), FEATURE_TYPE_SEQUENCE, INTENT, "test")
                    ],
                )
            ],
            False,
        ),
    ],
)
def test_check_labels_features_exist(messages, expected):
    attribute = TEXT
    classifier = DIETClassifier()
    assert classifier._check_labels_features_exist(messages, attribute) == expected


async def _train_persist_load_with_different_settings(
    pipeline, component_builder, tmp_path
):
    _config = RasaNLUModelConfig({"pipeline": pipeline, "language": "en"})

    (trainer, trained, persisted_path) = await train(
        _config,
        path=str(tmp_path),
        data="data/examples/rasa/demo-rasa-multi-intent.md",
        component_builder=component_builder,
    )

    assert trainer.pipeline
    assert trained.pipeline

    loaded = Interpreter.load(persisted_path, component_builder)

    assert loaded.pipeline
    assert loaded.parse("Rasa is great!") == trained.parse("Rasa is great!")


@pytest.mark.skip_on_windows
async def test_train_persist_load_with_different_settings_non_windows(
    component_builder, tmpdir
):
    pipeline = [
        {
            "name": "WhitespaceTokenizer",
            "intent_tokenization_flag": True,
            "intent_split_symbol": "+",
        },
        {"name": "CountVectorsFeaturizer"},
        {"name": "DIETClassifier", MASKED_LM: True, EPOCHS: 1},
    ]
    await _train_persist_load_with_different_settings(
        pipeline, component_builder, tmpdir
    )


async def test_train_persist_load_with_different_settings(component_builder, tmpdir):
    pipeline = [
        {"name": "WhitespaceTokenizer"},
        {"name": "CountVectorsFeaturizer"},
        {"name": "DIETClassifier", LOSS_TYPE: "margin", EPOCHS: 1},
    ]
    await _train_persist_load_with_different_settings(
        pipeline, component_builder, tmpdir
    )


async def test_train_persist_load_with_only_entity_recognition(
    component_builder, tmpdir
):
    pipeline = [
        {"name": "WhitespaceTokenizer"},
        {"name": "CountVectorsFeaturizer"},
        {
            "name": "DIETClassifier",
            ENTITY_RECOGNITION: True,
            INTENT_CLASSIFICATION: False,
            EPOCHS: 1,
        },
    ]
    await _train_persist_load_with_different_settings(
        pipeline, component_builder, tmpdir
    )


async def test_train_persist_load_with_only_intent_classification(
    component_builder, tmpdir
):
    pipeline = [
        {"name": "WhitespaceTokenizer"},
        {"name": "CountVectorsFeaturizer"},
        {
            "name": "DIETClassifier",
            ENTITY_RECOGNITION: False,
            INTENT_CLASSIFICATION: True,
            EPOCHS: 1,
        },
    ]
    await _train_persist_load_with_different_settings(
        pipeline, component_builder, tmpdir
    )


async def test_raise_error_on_incorrect_pipeline(component_builder, tmp_path: Path):
    _config = RasaNLUModelConfig(
        {
            "pipeline": [
                {"name": "WhitespaceTokenizer"},
                {"name": "DIETClassifier", EPOCHS: 1},
            ],
            "language": "en",
        }
    )

    with pytest.raises(Exception) as e:
        await train(
            _config,
            path=str(tmp_path),
            data=DEFAULT_DATA_PATH,
            component_builder=component_builder,
        )

    assert "'DIETClassifier' requires 'Featurizer'" in str(e.value)


def as_pipeline(*components):
    return [{"name": c} for c in components]


@pytest.mark.parametrize(
    "classifier_params, data_path, output_length, output_should_sum_to_1",
    [
        (
            {RANDOM_SEED: 42, EPOCHS: 1},
            "data/test/many_intents.md",
            10,
            True,
        ),  # default config
        (
            {RANDOM_SEED: 42, RANKING_LENGTH: 0, EPOCHS: 1},
            "data/test/many_intents.md",
            LABEL_RANKING_LENGTH,
            False,
        ),  # no normalization
        (
            {RANDOM_SEED: 42, RANKING_LENGTH: 3, EPOCHS: 1},
            "data/test/many_intents.md",
            3,
            True,
        ),  # lower than default ranking_length
        (
            {RANDOM_SEED: 42, RANKING_LENGTH: 12, EPOCHS: 1},
            "data/test/many_intents.md",
            LABEL_RANKING_LENGTH,
            False,
        ),  # higher than default ranking_length
        (
            {RANDOM_SEED: 42, EPOCHS: 1},
            DEFAULT_NLU_DATA,
            7,
            True,
        ),  # less intents than default ranking_length
    ],
)
async def test_softmax_normalization(
    component_builder,
    tmp_path,
    classifier_params,
    data_path,
    output_length,
    output_should_sum_to_1,
):
    pipeline = as_pipeline(
        "WhitespaceTokenizer", "CountVectorsFeaturizer", "DIETClassifier"
    )
    assert pipeline[2]["name"] == "DIETClassifier"
    pipeline[2].update(classifier_params)

    _config = RasaNLUModelConfig({"pipeline": pipeline})
    (trained_model, _, persisted_path) = await train(
        _config, path=str(tmp_path), data=data_path, component_builder=component_builder
    )
    loaded = Interpreter.load(persisted_path, component_builder)

    parse_data = loaded.parse("hello")
    intent_ranking = parse_data.get("intent_ranking")
    # check that the output was correctly truncated after normalization
    assert len(intent_ranking) == output_length

    # check whether normalization had the expected effect
    output_sums_to_1 = sum(
        [intent.get("confidence") for intent in intent_ranking]
    ) == pytest.approx(1)
    assert output_sums_to_1 == output_should_sum_to_1

    # check whether the normalization of rankings is reflected in intent prediction
    assert parse_data.get("intent") == intent_ranking[0]


@pytest.mark.parametrize(
    "classifier_params, output_length",
    [({LOSS_TYPE: "margin", RANDOM_SEED: 42, EPOCHS: 1}, LABEL_RANKING_LENGTH)],
)
async def test_margin_loss_is_not_normalized(
    monkeypatch, component_builder, tmpdir, classifier_params, output_length
):
    pipeline = as_pipeline(
        "WhitespaceTokenizer", "CountVectorsFeaturizer", "DIETClassifier"
    )
    assert pipeline[2]["name"] == "DIETClassifier"
    pipeline[2].update(classifier_params)

    mock = Mock()
    monkeypatch.setattr(train_utils, "normalize", mock.normalize)

    _config = RasaNLUModelConfig({"pipeline": pipeline})
    (trained_model, _, persisted_path) = await train(
        _config,
        path=tmpdir.strpath,
        data="data/test/many_intents.md",
        component_builder=component_builder,
    )
    loaded = Interpreter.load(persisted_path, component_builder)

    parse_data = loaded.parse("hello")
    intent_ranking = parse_data.get("intent_ranking")

    # check that the output was not normalized
    mock.normalize.assert_not_called()

    # check that the output was correctly truncated
    assert len(intent_ranking) == output_length

    # make sure top ranking is reflected in intent prediction
    assert parse_data.get("intent") == intent_ranking[0]


async def test_set_random_seed(component_builder, tmpdir):
    """test if train result is the same for two runs of tf embedding"""

    # set fixed random seed
    _config = RasaNLUModelConfig(
        {
            "pipeline": [
                {"name": "WhitespaceTokenizer"},
                {"name": "CountVectorsFeaturizer"},
                {"name": "DIETClassifier", RANDOM_SEED: 1, EPOCHS: 1},
            ],
            "language": "en",
        }
    )

    # first run
    (trained_a, _, persisted_path_a) = await train(
        _config,
        path=tmpdir.strpath + "_a",
        data=DEFAULT_DATA_PATH,
        component_builder=component_builder,
    )
    # second run
    (trained_b, _, persisted_path_b) = await train(
        _config,
        path=tmpdir.strpath + "_b",
        data=DEFAULT_DATA_PATH,
        component_builder=component_builder,
    )

    loaded_a = Interpreter.load(persisted_path_a, component_builder)
    loaded_b = Interpreter.load(persisted_path_b, component_builder)
    result_a = loaded_a.parse("hello")["intent"]["confidence"]
    result_b = loaded_b.parse("hello")["intent"]["confidence"]

    assert result_a == result_b


@pytest.mark.parametrize(
    "log_level", ["epoch", "batch", "minibatch",],
)
async def test_train_tensorboard_logging(
    log_level: Text, component_builder: ComponentBuilder, tmpdir: Path
):
    tensorboard_log_dir = Path(tmpdir / "tensorboard")

    assert not tensorboard_log_dir.exists()

    _config = RasaNLUModelConfig(
        {
            "pipeline": [
                {"name": "WhitespaceTokenizer"},
                {"name": "CountVectorsFeaturizer"},
                {
                    "name": "DIETClassifier",
                    EPOCHS: 1,
                    TENSORBOARD_LOG_LEVEL: log_level,
                    TENSORBOARD_LOG_DIR: str(tensorboard_log_dir),
                    EVAL_NUM_EXAMPLES: 15,
                    EVAL_NUM_EPOCHS: 1,
                },
            ],
            "language": "en",
        }
    )

    await train(
        _config,
        path=str(tmpdir),
        data="data/examples/rasa/demo-rasa-multi-intent.md",
        component_builder=component_builder,
    )

    assert tensorboard_log_dir.exists()

    all_files = list(tensorboard_log_dir.rglob("*.*"))
    assert len(all_files) == 10


async def test_train_model_checkpointing(
    component_builder: ComponentBuilder, tmpdir: Path
):
    model_name = "nlu-checkpointed-model"
    best_model_file = Path(str(tmpdir), model_name)
    assert not best_model_file.exists()

    _config = RasaNLUModelConfig(
        {
            "pipeline": [
                {"name": "WhitespaceTokenizer"},
                {"name": "CountVectorsFeaturizer"},
                {
                    "name": "DIETClassifier",
                    EPOCHS: 5,
                    EVAL_NUM_EXAMPLES: 10,
                    EVAL_NUM_EPOCHS: 1,
                    CHECKPOINT_MODEL: True,
                },
            ],
            "language": "en",
        }
    )

    await train(
        _config,
        path=str(tmpdir),
        data="data/examples/rasa/demo-rasa.md",
        component_builder=component_builder,
        fixed_model_name=model_name,
    )

    assert best_model_file.exists()

    """
    Tricky to validate the *exact* number of files that should be there, however there must be at least the following:
        - metadata.json
        - checkpoint
        - component_1_CountVectorsFeaturizer (as per the pipeline above)
        - component_2_DIETClassifier files (more than 1 file)
    """
    all_files = list(best_model_file.rglob("*.*"))
    assert len(all_files) > 4


@pytest.mark.parametrize(
    "classifier_params",
    [
        {RANDOM_SEED: 1, EPOCHS: 1, BILOU_FLAG: False},
        {RANDOM_SEED: 1, EPOCHS: 1, BILOU_FLAG: True},
    ],
)
async def test_train_persist_load_with_composite_entities(
    classifier_params, component_builder, tmpdir
):
    pipeline = as_pipeline(
        "WhitespaceTokenizer", "CountVectorsFeaturizer", "DIETClassifier"
    )
    assert pipeline[2]["name"] == "DIETClassifier"
    pipeline[2].update(classifier_params)

    _config = RasaNLUModelConfig({"pipeline": pipeline, "language": "en"})

    (trainer, trained, persisted_path) = await train(
        _config,
        path=tmpdir.strpath,
        data="data/test/demo-rasa-composite-entities.md",
        component_builder=component_builder,
    )

    assert trainer.pipeline
    assert trained.pipeline

    loaded = Interpreter.load(persisted_path, component_builder)

    assert loaded.pipeline
    text = "I am looking for an italian restaurant"
    assert loaded.parse(text) == trained.parse(text)
