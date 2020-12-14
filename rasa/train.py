import asyncio
import os
import tempfile
from contextlib import ExitStack
from typing import Text, NamedTuple, Tuple, Optional, List, Union, Dict

import rasa.core.interpreter
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter
from rasa.shared.importers.importer import TrainingDataImporter
from rasa import model, telemetry
from rasa.model import FingerprintComparisonResult
from rasa.shared.core.domain import Domain
import rasa.shared.utils.common
from rasa.nlu.model import Interpreter
import rasa.utils.common
from rasa.utils.common import TempDirectoryPath

from rasa.shared.utils.cli import (
    print_success,
    print_warning,
    print_error,
    print_color,
)
import rasa.shared.exceptions
import rasa.shared.utils.io
from rasa.shared.constants import (
    DEFAULT_MODELS_PATH,
    DEFAULT_CORE_SUBDIRECTORY_NAME,
    DEFAULT_NLU_SUBDIRECTORY_NAME,
)

CODE_CORE_NEEDS_TO_BE_RETRAINED = 0b0001
CODE_NLU_NEEDS_TO_BE_RETRAINED = 0b0010
CODE_NLG_NEEDS_TO_BE_RETRAINED = 0b0100
CODE_FORCED_TRAINING = 0b1000


class TrainingResult(NamedTuple):
    """Holds information about the results of training."""

    model: Optional[Text] = None
    code: int = 0


def train(
    domain: Text,
    config: Text,
    training_files: Union[Text, List[Text]],
    output: Text = DEFAULT_MODELS_PATH,
    dry_run: bool = False,
    force_training: bool = False,
    fixed_model_name: Optional[Text] = None,
    persist_nlu_training_data: bool = False,
    core_additional_arguments: Optional[Dict] = None,
    nlu_additional_arguments: Optional[Dict] = None,
    loop: Optional[asyncio.AbstractEventLoop] = None,
) -> TrainingResult:
    """Runs Rasa Core and NLU training in `async` loop.

    Args:
        domain: Path to the domain file.
        config: Path to the config for Core and NLU.
        training_files: Paths to the training data for Core and NLU.
        output: Output path.
        dry_run: If `True` then no training will be done, and the information about
            whether the training needs to be done will be printed.
        force_training: If `True` retrain model even if data has not changed.
        fixed_model_name: Name of model to be stored.
        persist_nlu_training_data: `True` if the NLU training data should be persisted
            with the model.
        core_additional_arguments: Additional training parameters for core training.
        nlu_additional_arguments: Additional training parameters forwarded to training
            method of each NLU component.
        loop: The event loop which will be used to run `async` functions.

    Returns:
        An instance of `TrainingResult`.
    """
    return rasa.utils.common.run_in_loop(
        train_async(
            domain=domain,
            config=config,
            training_files=training_files,
            output=output,
            dry_run=dry_run,
            force_training=force_training,
            fixed_model_name=fixed_model_name,
            persist_nlu_training_data=persist_nlu_training_data,
            core_additional_arguments=core_additional_arguments,
            nlu_additional_arguments=nlu_additional_arguments,
        ),
        loop,
    )


async def train_async(
    domain: Union[Domain, Text],
    config: Text,
    training_files: Optional[Union[Text, List[Text]]],
    output: Text = DEFAULT_MODELS_PATH,
    dry_run: bool = False,
    force_training: bool = False,
    fixed_model_name: Optional[Text] = None,
    persist_nlu_training_data: bool = False,
    core_additional_arguments: Optional[Dict] = None,
    nlu_additional_arguments: Optional[Dict] = None,
) -> TrainingResult:
    """Trains a Rasa model (Core and NLU).

    Args:
        domain: Path to the domain file.
        config: Path to the config for Core and NLU.
        training_files: Paths to the training data for Core and NLU.
        output_path: Output path.
        dry_run: If `True` then no training will be done, and the information about
            whether the training needs to be done will be printed.
        force_training: If `True` retrain model even if data has not changed.
        fixed_model_name: Name of model to be stored.
        persist_nlu_training_data: `True` if the NLU training data should be persisted
            with the model.
        core_additional_arguments: Additional training parameters for core training.
        nlu_additional_arguments: Additional training parameters forwarded to training
            method of each NLU component.

    Returns:
        An instance of `TrainingResult`.
    """
    file_importer = TrainingDataImporter.load_from_config(
        config, domain, training_files
    )
    with ExitStack() as stack:
        train_path = stack.enter_context(TempDirectoryPath(tempfile.mkdtemp()))

        domain = await file_importer.get_domain()

        if domain.is_empty():
            nlu_model = await handle_domain_if_not_exists(
                file_importer, output, fixed_model_name
            )
            return TrainingResult(model=nlu_model)

        return await _train_async_internal(
            file_importer,
            train_path,
            output,
            dry_run,
            force_training,
            fixed_model_name,
            persist_nlu_training_data,
            core_additional_arguments=core_additional_arguments,
            nlu_additional_arguments=nlu_additional_arguments,
        )


async def handle_domain_if_not_exists(
    file_importer: TrainingDataImporter, output_path, fixed_model_name
):
    nlu_model_only = await _train_nlu_with_validated_data(
        file_importer, output=output_path, fixed_model_name=fixed_model_name
    )
    print_warning(
        "Core training was skipped because no valid domain file was found. "
        "Only an NLU-model was created. Please specify a valid domain using "
        "the '--domain' argument or check if the provided domain file exists."
    )
    return nlu_model_only


def dry_run_result(
    fingerprint_comparison: FingerprintComparisonResult,
) -> Tuple[int, List[Text]]:
    """Returns a dry run result.

    Args:
        fingerprint_comparison: A result of fingerprint comparison operation.

    Returns:
        A tuple where the first element is the result code and the second
        is the list of human-readable texts that need to be printed to the end user.
    """
    code = 0
    texts = []

    if fingerprint_comparison.force_training:
        code = CODE_FORCED_TRAINING
        texts.append("The training was forced.")
        return code, texts

    if fingerprint_comparison.core:
        code += CODE_CORE_NEEDS_TO_BE_RETRAINED
        texts.append("Core model should be retrained.")

    if fingerprint_comparison.nlu:
        code += CODE_NLU_NEEDS_TO_BE_RETRAINED
        texts.append("NLU model should be retrained.")

    if fingerprint_comparison.nlg:
        code += CODE_NLG_NEEDS_TO_BE_RETRAINED
        texts.append("Responses in the domain should be updated.")

    if code == 0:
        texts.append("No training required.")

    return code, texts


async def _train_async_internal(
    file_importer: TrainingDataImporter,
    train_path: Text,
    output_path: Text,
    dry_run: bool,
    force_training: bool,
    fixed_model_name: Optional[Text],
    persist_nlu_training_data: bool,
    core_additional_arguments: Optional[Dict] = None,
    nlu_additional_arguments: Optional[Dict] = None,
) -> TrainingResult:
    """Trains a Rasa model (Core and NLU). Use only from `train_async`.

    Args:
        file_importer: `TrainingDataImporter` which supplies the training data.
        train_path: Directory in which to train the model.
        output_path: Output path.
        dry_run: If `True` then no training will be done, and the information about
            whether the training needs to be done will be printed.
        force_training: If `True` retrain model even if data has not changed.
        fixed_model_name: Name of model to be stored.
        persist_nlu_training_data: `True` if the NLU training data should be persisted
            with the model.
        core_additional_arguments: Additional training parameters for core training.
        nlu_additional_arguments: Additional training parameters forwarded to training
            method of each NLU component.

    Returns:
        An instance of `TrainingResult`.
    """
    stories, nlu_data = await asyncio.gather(
        file_importer.get_stories(), file_importer.get_nlu_data()
    )

    new_fingerprint = await model.model_fingerprint(file_importer)
    old_model = model.get_latest_model(output_path)

    fingerprint_comparison = model.should_retrain(
        new_fingerprint, old_model, train_path, force_training
    )

    if dry_run:
        code, texts = dry_run_result(fingerprint_comparison)
        for text in texts:
            print_warning(text) if code > 0 else print_success(text)
        return TrainingResult(code=code)

    if nlu_data.has_e2e_examples():
        rasa.shared.utils.common.mark_as_experimental_feature("end-to-end training")

    if stories.is_empty() and nlu_data.contains_no_pure_nlu_data():
        print_error(
            "No training data given. Please provide stories and NLU data in "
            "order to train a Rasa model using the '--data' argument."
        )
        return TrainingResult()

    if stories.is_empty():
        print_warning("No stories present. Just a Rasa NLU model will be trained.")
        trained_model = await _train_nlu_with_validated_data(
            file_importer,
            output=output_path,
            fixed_model_name=fixed_model_name,
            persist_nlu_training_data=persist_nlu_training_data,
            additional_arguments=nlu_additional_arguments,
        )
        return TrainingResult(model=trained_model)

    # We will train nlu if there are any nlu example, including from e2e stories.
    if nlu_data.contains_no_pure_nlu_data() and not nlu_data.has_e2e_examples():
        print_warning("No NLU data present. Just a Rasa Core model will be trained.")
        trained_model = await _train_core_with_validated_data(
            file_importer,
            output=output_path,
            fixed_model_name=fixed_model_name,
            additional_arguments=core_additional_arguments,
        )

        return TrainingResult(model=trained_model)

    new_fingerprint = await model.model_fingerprint(file_importer)
    old_model = model.get_latest_model(output_path)

    if not force_training:
        fingerprint_comparison = model.should_retrain(
            new_fingerprint,
            old_model,
            train_path,
            has_e2e_examples=nlu_data.has_e2e_examples(),
        )
    else:
        fingerprint_comparison = FingerprintComparisonResult(force_training=True)

    if fingerprint_comparison.is_training_required():
        async with telemetry.track_model_training(file_importer, model_type="rasa"):
            await _do_training(
                file_importer,
                output_path=output_path,
                train_path=train_path,
                fingerprint_comparison_result=fingerprint_comparison,
                fixed_model_name=fixed_model_name,
                persist_nlu_training_data=persist_nlu_training_data,
                core_additional_arguments=core_additional_arguments,
                nlu_additional_arguments=nlu_additional_arguments,
                old_model_zip_path=old_model,
            )
        trained_model = model.package_model(
            fingerprint=new_fingerprint,
            output_directory=output_path,
            train_path=train_path,
            fixed_model_name=fixed_model_name,
        )
        return TrainingResult(model=trained_model)

    print_success(
        "Nothing changed. You can use the old model stored at '{}'."
        "".format(os.path.abspath(old_model))
    )
    return TrainingResult(model=old_model)


async def _do_training(
    file_importer: TrainingDataImporter,
    output_path: Text,
    train_path: Text,
    fingerprint_comparison_result: Optional[FingerprintComparisonResult] = None,
    fixed_model_name: Optional[Text] = None,
    persist_nlu_training_data: bool = False,
    core_additional_arguments: Optional[Dict] = None,
    nlu_additional_arguments: Optional[Dict] = None,
    old_model_zip_path: Optional[Text] = None,
):
    if not fingerprint_comparison_result:
        fingerprint_comparison_result = FingerprintComparisonResult()

    interpreter_path = None
    if fingerprint_comparison_result.should_retrain_nlu():
        model_path = await _train_nlu_with_validated_data(
            file_importer,
            output=output_path,
            train_path=train_path,
            fixed_model_name=fixed_model_name,
            persist_nlu_training_data=persist_nlu_training_data,
            additional_arguments=nlu_additional_arguments,
        )
        interpreter_path = os.path.join(model_path, DEFAULT_NLU_SUBDIRECTORY_NAME)
    else:
        print_color(
            "NLU data/configuration did not change. No need to retrain NLU model.",
            color=rasa.shared.utils.io.bcolors.OKBLUE,
        )

    if fingerprint_comparison_result.should_retrain_core():
        await _train_core_with_validated_data(
            file_importer,
            output=output_path,
            train_path=train_path,
            fixed_model_name=fixed_model_name,
            additional_arguments=core_additional_arguments,
            interpreter=_load_interpreter(interpreter_path)
            or _interpreter_from_previous_model(old_model_zip_path),
        )
    elif fingerprint_comparison_result.should_retrain_nlg():
        print_color(
            "Core stories/configuration did not change. "
            "Only the templates section has been changed. A new model with "
            "the updated templates will be created.",
            color=rasa.shared.utils.io.bcolors.OKBLUE,
        )
        await model.update_model_with_new_domain(file_importer, train_path)
    else:
        print_color(
            "Core stories/configuration did not change. No need to retrain Core model.",
            color=rasa.shared.utils.io.bcolors.OKBLUE,
        )


def _load_interpreter(
    interpreter_path: Optional[Text],
) -> Optional[NaturalLanguageInterpreter]:
    if interpreter_path:
        return rasa.core.interpreter.create_interpreter(interpreter_path)

    return None


def _interpreter_from_previous_model(
    old_model_zip_path: Optional[Text],
) -> Optional[NaturalLanguageInterpreter]:
    if not old_model_zip_path:
        return None

    with model.unpack_model(old_model_zip_path) as unpacked:
        _, old_nlu = model.get_model_subdirectories(unpacked)
        return rasa.core.interpreter.create_interpreter(old_nlu)


def train_core(
    domain: Union[Domain, Text],
    config: Text,
    stories: Text,
    output: Text,
    train_path: Optional[Text] = None,
    fixed_model_name: Optional[Text] = None,
    additional_arguments: Optional[Dict] = None,
) -> Optional[Text]:
    return rasa.utils.common.run_in_loop(
        train_core_async(
            domain=domain,
            config=config,
            stories=stories,
            output=output,
            train_path=train_path,
            fixed_model_name=fixed_model_name,
            additional_arguments=additional_arguments,
        )
    )


async def train_core_async(
    domain: Union[Domain, Text],
    config: Text,
    stories: Text,
    output: Text,
    train_path: Optional[Text] = None,
    fixed_model_name: Optional[Text] = None,
    additional_arguments: Optional[Dict] = None,
) -> Optional[Text]:
    """Trains a Core model.

    Args:
        domain: Path to the domain file.
        config: Path to the config file for Core.
        stories: Path to the Core training data.
        output: Output path.
        train_path: If `None` the model will be trained in a temporary
            directory, otherwise in the provided directory.
        fixed_model_name: Name of model to be stored.
        additional_arguments: Additional training parameters.

    Returns:
        If `train_path` is given it returns the path to the model archive,
        otherwise the path to the directory with the trained model files.

    """

    file_importer = TrainingDataImporter.load_core_importer_from_config(
        config, domain, [stories]
    )
    stories, nlu_data, domain = await asyncio.gather(
        file_importer.get_stories(),
        file_importer.get_nlu_data(),
        file_importer.get_domain(),
    )

    if nlu_data.has_e2e_examples():
        print_error(
            "Stories file contains e2e stories. Please train using `rasa train` so that"
            " the NLU model is also trained."
        )
        return None

    if domain.is_empty():
        print_error(
            "Core training was skipped because no valid domain file was found. "
            "Please specify a valid domain using '--domain' argument or check "
            "if the provided domain file exists."
        )
        return None

    if not stories:
        print_error(
            "No stories given. Please provide stories in order to "
            "train a Rasa Core model using the '--stories' argument."
        )
        return

    return await _train_core_with_validated_data(
        file_importer,
        output=output,
        train_path=train_path,
        fixed_model_name=fixed_model_name,
        additional_arguments=additional_arguments,
    )


async def _train_core_with_validated_data(
    file_importer: TrainingDataImporter,
    output: Text,
    train_path: Optional[Text] = None,
    fixed_model_name: Optional[Text] = None,
    additional_arguments: Optional[Dict] = None,
    interpreter: Optional[Interpreter] = None,
) -> Optional[Text]:
    """Train Core with validated training and config data."""

    import rasa.core.train

    with ExitStack() as stack:
        if train_path:
            # If the train path was provided, do nothing on exit.
            _train_path = train_path
        else:
            # Otherwise, create a temp train path and clean it up on exit.
            _train_path = stack.enter_context(TempDirectoryPath(tempfile.mkdtemp()))

        # normal (not compare) training
        print_color("Training Core model...", color=rasa.shared.utils.io.bcolors.OKBLUE)
        domain, config = await asyncio.gather(
            file_importer.get_domain(), file_importer.get_config()
        )
        async with telemetry.track_model_training(file_importer, model_type="core"):
            await rasa.core.train(
                domain_file=domain,
                training_resource=file_importer,
                output_path=os.path.join(_train_path, DEFAULT_CORE_SUBDIRECTORY_NAME),
                policy_config=config,
                additional_arguments=additional_arguments,
                interpreter=interpreter,
            )
        print_color(
            "Core model training completed.", color=rasa.shared.utils.io.bcolors.OKBLUE
        )

        if train_path is None:
            # Only Core was trained.
            new_fingerprint = await model.model_fingerprint(file_importer)
            return model.package_model(
                fingerprint=new_fingerprint,
                output_directory=output,
                train_path=_train_path,
                fixed_model_name=fixed_model_name,
                model_prefix="core-",
            )

        return _train_path


def train_nlu(
    config: Text,
    nlu_data: Text,
    output: Text,
    train_path: Optional[Text] = None,
    fixed_model_name: Optional[Text] = None,
    persist_nlu_training_data: bool = False,
    additional_arguments: Optional[Dict] = None,
    domain: Optional[Union[Domain, Text]] = None,
) -> Optional[Text]:
    """Trains an NLU model.

    Args:
        config: Path to the config file for NLU.
        nlu_data: Path to the NLU training data.
        output: Output path.
        train_path: If `None` the model will be trained in a temporary
            directory, otherwise in the provided directory.
        fixed_model_name: Name of the model to be stored.
        persist_nlu_training_data: `True` if the NLU training data should be persisted
                                   with the model.
        additional_arguments: Additional training parameters which will be passed to
                              the `train` method of each component.
        domain: Path to the optional domain file/Domain object.


    Returns:
        If `train_path` is given it returns the path to the model archive,
        otherwise the path to the directory with the trained model files.

    """

    return rasa.utils.common.run_in_loop(
        _train_nlu_async(
            config,
            nlu_data,
            output,
            train_path,
            fixed_model_name,
            persist_nlu_training_data,
            additional_arguments,
            domain=domain,
        )
    )


async def _train_nlu_async(
    config: Text,
    nlu_data: Text,
    output: Text,
    train_path: Optional[Text] = None,
    fixed_model_name: Optional[Text] = None,
    persist_nlu_training_data: bool = False,
    additional_arguments: Optional[Dict] = None,
    domain: Optional[Union[Domain, Text]] = None,
) -> Optional[Text]:
    if not nlu_data:
        print_error(
            "No NLU data given. Please provide NLU data in order to train "
            "a Rasa NLU model using the '--nlu' argument."
        )
        return

    # training NLU only hence the training files still have to be selected
    file_importer = TrainingDataImporter.load_nlu_importer_from_config(
        config, domain, training_data_paths=[nlu_data]
    )

    training_data = await file_importer.get_nlu_data()
    if training_data.contains_no_pure_nlu_data():
        print_error(
            f"Path '{nlu_data}' doesn't contain valid NLU data in it. "
            f"Please verify the data format. "
            f"The NLU model training will be skipped now."
        )
        return

    return await _train_nlu_with_validated_data(
        file_importer,
        output=output,
        train_path=train_path,
        fixed_model_name=fixed_model_name,
        persist_nlu_training_data=persist_nlu_training_data,
        additional_arguments=additional_arguments,
    )


async def _train_nlu_with_validated_data(
    file_importer: TrainingDataImporter,
    output: Text,
    train_path: Optional[Text] = None,
    fixed_model_name: Optional[Text] = None,
    persist_nlu_training_data: bool = False,
    additional_arguments: Optional[Dict] = None,
) -> Optional[Text]:
    """Train NLU with validated training and config data."""

    import rasa.nlu.train

    if additional_arguments is None:
        additional_arguments = {}

    with ExitStack() as stack:
        if train_path:
            # If the train path was provided, do nothing on exit.
            _train_path = train_path
        else:
            # Otherwise, create a temp train path and clean it up on exit.
            _train_path = stack.enter_context(TempDirectoryPath(tempfile.mkdtemp()))
        config = await file_importer.get_config()
        print_color("Training NLU model...", color=rasa.shared.utils.io.bcolors.OKBLUE)
        async with telemetry.track_model_training(file_importer, model_type="nlu"):
            await rasa.nlu.train(
                config,
                file_importer,
                _train_path,
                fixed_model_name="nlu",
                persist_nlu_training_data=persist_nlu_training_data,
                **additional_arguments,
            )
        print_color(
            "NLU model training completed.", color=rasa.shared.utils.io.bcolors.OKBLUE
        )

        if train_path is None:
            # Only NLU was trained
            new_fingerprint = await model.model_fingerprint(file_importer)

            return model.package_model(
                fingerprint=new_fingerprint,
                output_directory=output,
                train_path=_train_path,
                fixed_model_name=fixed_model_name,
                model_prefix="nlu-",
            )

        return _train_path
