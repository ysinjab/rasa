import os
from pathlib import Path
from typing import Text
import pytest

from rasa.core.training.converters.story_markdown_to_yaml_converter import (
    StoryMarkdownToYamlConverter,
)

from rasa.shared.constants import LATEST_TRAINING_DATA_FORMAT_VERSION


@pytest.mark.parametrize(
    "training_data_file, should_filter",
    [
        ("data/test_stories/stories.md", True),
        ("data/test_nlu/default_retrieval_intents.md", False),
        ("data/test_yaml_stories/rules_without_stories.yml", False),
    ],
)
def test_converter_filters_correct_files(training_data_file: Text, should_filter: bool):
    assert should_filter == StoryMarkdownToYamlConverter.filter(
        Path(training_data_file)
    )


async def test_stories_are_converted(tmp_path: Path):
    converted_data_folder = tmp_path / "converted_data"
    converted_data_folder.mkdir()

    training_data_folder = tmp_path / "data" / "core"
    training_data_folder.mkdir(parents=True)
    training_data_file = Path(training_data_folder / "stories.md")

    simple_story_md = """
    ## happy path
    * greet OR goodbye
        - utter_greet
        - form{"name": null}
        - slot{"name": ["value1", "value2"]}
    """

    training_data_file.write_text(simple_story_md)

    with pytest.warns(None) as warnings:
        await StoryMarkdownToYamlConverter().convert_and_write(
            training_data_file, converted_data_folder
        )

    assert not warnings

    assert len(os.listdir(converted_data_folder)) == 1

    with open(f"{converted_data_folder}/stories_converted.yml", "r") as f:
        content = f.read()
        assert content == (
            f'version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"\n'
            "stories:\n"
            "- story: happy path\n"
            "  steps:\n"
            "  - or:\n"
            "    - intent: greet\n"
            "    - intent: goodbye\n"
            "  - action: utter_greet\n"
            "  - active_loop: null\n"
            "  - slot_was_set:\n"
            "    - name:\n"
            "      - value1\n"
            "      - value2\n"
        )


async def test_test_stories(tmp_path: Path):
    converted_data_folder = tmp_path / "converted_data"
    converted_data_folder.mkdir()

    test_data_folder = tmp_path / "tests"
    test_data_folder.mkdir(exist_ok=True)
    test_data_file = Path(test_data_folder / "test_stories.md")

    simple_story_md = """
    ## ask product
    * faq: what is [Rasa X](product)?
        - slot{"product": "x"}
        - respond_faq
        - action_set_faq_slot
    """

    test_data_file.write_text(simple_story_md)

    with pytest.warns(None) as warnings:
        await StoryMarkdownToYamlConverter().convert_and_write(
            test_data_file, converted_data_folder
        )

    assert not warnings

    assert len(list(converted_data_folder.glob("*"))) == 1

    with open(f"{converted_data_folder}/test_stories_converted.yml", "r") as f:
        content = f.read()
        assert content == (
            f'version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"\n'
            "stories:\n"
            "- story: ask product\n"
            "  steps:\n"
            "  - intent: faq\n"
            "    user: |-\n"
            "      what is [Rasa X](product)?\n"
            "  - slot_was_set:\n"
            "    - product: x\n"
            "  - action: respond_faq\n"
            "  - action: action_set_faq_slot\n"
        )


async def test_test_stories_conversion_response_key(tmp_path: Path):
    converted_data_folder = tmp_path / "converted_data"
    converted_data_folder.mkdir()

    test_data_folder = tmp_path / "tests"
    test_data_folder.mkdir(exist_ok=True)
    test_data_file = Path(test_data_folder / "test_stories.md")

    simple_story_md = """
    ## id

    * out_of_scope/other: hahaha
        - utter_out_of_scope/other
    """

    test_data_file.write_text(simple_story_md)

    await StoryMarkdownToYamlConverter().convert_and_write(
        test_data_file, converted_data_folder
    )

    assert len(os.listdir(converted_data_folder)) == 1
    with open(f"{converted_data_folder}/test_stories_converted.yml", "r") as f:
        content = f.read()
        assert content == (
            f'version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"\n'
            "stories:\n"
            "- story: id\n"
            "  steps:\n"
            "  - intent: out_of_scope/other\n"
            "    user: |-\n"
            "      hahaha\n"
            "  - action: utter_out_of_scope/other\n"
        )


async def test_stories_conversion_response_key(tmp_path: Path):
    converted_data_folder = tmp_path / "converted_data"
    converted_data_folder.mkdir()

    training_data_folder = tmp_path / "data" / "core"
    training_data_folder.mkdir(parents=True)
    training_data_file = Path(training_data_folder / "stories.md")

    simple_story_md = """
    ## id
    * out_of_scope/other
        - utter_out_of_scope/other
    """

    training_data_file.write_text(simple_story_md)

    await StoryMarkdownToYamlConverter().convert_and_write(
        training_data_file, converted_data_folder
    )

    assert len(os.listdir(converted_data_folder)) == 1

    with open(f"{converted_data_folder}/stories_converted.yml", "r") as f:
        content = f.read()
        assert content == (
            f'version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"\n'
            "stories:\n"
            "- story: id\n"
            "  steps:\n"
            "  - intent: out_of_scope/other\n"
            "  - action: utter_out_of_scope/other\n"
        )
