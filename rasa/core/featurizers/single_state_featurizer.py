import logging
import numpy as np
import scipy.sparse
from typing import List, Optional, Dict, Text, Set, Any
from collections import defaultdict

import rasa.shared.utils.io
from rasa.nlu.constants import TOKENS_NAMES
from rasa.shared.core.domain import SubState, State, Domain
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter, RegexInterpreter
from rasa.shared.core.constants import PREVIOUS_ACTION, ACTIVE_LOOP, USER, SLOTS
from rasa.shared.constants import DOCS_URL_MIGRATION_GUIDE
from rasa.shared.core.trackers import is_prev_action_listen_in_state
from rasa.shared.nlu.constants import (
    ENTITIES,
    FEATURE_TYPE_SENTENCE,
    ACTION_TEXT,
    ACTION_NAME,
    INTENT,
    TEXT,
    NO_ENTITY_TAG,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_TAGS,
)
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.training_data.message import Message
from rasa.utils.tensorflow.model_data_utils import TAG_ID_ORIGIN
from rasa.utils.tensorflow.constants import IDS

logger = logging.getLogger(__name__)


class SingleStateFeaturizer:
    """Base class to transform the dialogue state into an ML format.

    Subclasses of SingleStateFeaturizer will decide how a bot will
    transform the dialogue state into a dictionary mapping an attribute
    to its features. Possible attributes are: INTENT, TEXT, ACTION_NAME,
    ACTION_TEXT, ENTITIES, SLOTS and ACTIVE_LOOP. Each attribute will be
    featurized into a list of `rasa.utils.features.Features`.
    """

    def __init__(self) -> None:
        """Initialize the single state featurizer."""
        # rasa core can be trained separately, therefore interpreter during training
        # will be `RegexInterpreter`. If the model is combined with a rasa nlu model
        # during prediction the interpreter might be different.
        # If that is the case, we need to make sure to "reset" the interpreter.
        self._use_regex_interpreter = False
        self._default_feature_states = {}
        self.action_texts = []
        self.entity_tag_id_mapping = {}

    def get_entity_tag_ids(self) -> Dict[Text, int]:
        """Returns the tag to index mapping for entities.

        Returns:
            Tag to index mapping.
        """
        if ENTITIES not in self._default_feature_states:
            return {}

        tag_ids = {
            tag: idx + 1  # +1 to keep 0 for the NO_ENTITY_TAG
            for tag, idx in self._default_feature_states[ENTITIES].items()
        }
        tag_ids[NO_ENTITY_TAG] = 0
        return tag_ids

    def prepare_for_training(
        self, domain: Domain, interpreter: NaturalLanguageInterpreter
    ) -> None:
        """Gets necessary information for featurization from domain.

        Args:
            domain: An instance of :class:`rasa.shared.core.domain.Domain`.
            interpreter: The interpreter used to encode the state
        """
        if isinstance(interpreter, RegexInterpreter):
            # this method is called during training,
            # RegexInterpreter means that core was trained separately
            self._use_regex_interpreter = True

        # store feature states for each attribute in order to create binary features
        def convert_to_dict(feature_states: List[Text]) -> Dict[Text, int]:
            return {
                feature_state: idx for idx, feature_state in enumerate(feature_states)
            }

        self._default_feature_states[INTENT] = convert_to_dict(domain.intents)
        self._default_feature_states[ACTION_NAME] = convert_to_dict(
            domain.action_names_or_texts
        )
        self._default_feature_states[ENTITIES] = convert_to_dict(domain.entity_states)
        self._default_feature_states[SLOTS] = convert_to_dict(domain.slot_states)
        self._default_feature_states[ACTIVE_LOOP] = convert_to_dict(domain.form_names)
        self.action_texts = domain.action_texts
        self.entity_tag_id_mapping = self.get_entity_tag_ids()

    def _state_features_for_attribute(
        self, sub_state: SubState, attribute: Text
    ) -> Dict[Text, int]:
        if attribute in {INTENT, ACTION_NAME}:
            return {sub_state[attribute]: 1}
        elif attribute == ENTITIES:
            return {entity: 1 for entity in sub_state.get(ENTITIES, [])}
        elif attribute == ACTIVE_LOOP:
            return {sub_state["name"]: 1}
        elif attribute == SLOTS:
            return {
                f"{slot_name}_{i}": value
                for slot_name, slot_as_feature in sub_state.items()
                for i, value in enumerate(slot_as_feature)
            }
        else:
            raise ValueError(
                f"Given attribute '{attribute}' is not supported. "
                f"It must be one of '{self._default_feature_states.keys()}'."
            )

    def _create_features(
        self, sub_state: SubState, attribute: Text, sparse: bool = False
    ) -> List["Features"]:
        state_features = self._state_features_for_attribute(sub_state, attribute)

        features = np.zeros(len(self._default_feature_states[attribute]), np.float32)
        for state_feature, value in state_features.items():
            # check that the value is in default_feature_states to be able to assign
            # its value
            if state_feature in self._default_feature_states[attribute]:
                features[self._default_feature_states[attribute][state_feature]] = value
        features = np.expand_dims(features, 0)

        if sparse:
            features = scipy.sparse.coo_matrix(features)

        features = Features(
            features, FEATURE_TYPE_SENTENCE, attribute, self.__class__.__name__
        )
        return [features]

    @staticmethod
    def _to_sparse_sentence_features(
        sparse_sequence_features: List["Features"],
    ) -> List["Features"]:
        return [
            Features(
                scipy.sparse.coo_matrix(feature.features.sum(0)),
                FEATURE_TYPE_SENTENCE,
                feature.attribute,
                feature.origin,
            )
            for feature in sparse_sequence_features
        ]

    def _get_features_from_parsed_message(
        self, parsed_message: Optional[Message], attributes: Set[Text]
    ) -> Dict[Text, List["Features"]]:
        if parsed_message is None:
            return {}

        output = defaultdict(list)
        for attribute in attributes:
            all_features = parsed_message.get_sparse_features(
                attribute
            ) + parsed_message.get_dense_features(attribute)

            for features in all_features:
                if features is not None:
                    output[attribute].append(features)

        # if features for INTENT or ACTION_NAME exist,
        # they are always sparse sequence features;
        # transform them to sentence sparse features
        if output.get(INTENT):
            output[INTENT] = self._to_sparse_sentence_features(output[INTENT])
        if output.get(ACTION_NAME):
            output[ACTION_NAME] = self._to_sparse_sentence_features(output[ACTION_NAME])

        return output

    @staticmethod
    def _get_name_attribute(attributes: Set[Text]) -> Optional[Text]:
        # there is always either INTENT or ACTION_NAME
        return next(
            (
                attribute
                for attribute in attributes
                if attribute in {INTENT, ACTION_NAME}
            ),
            None,
        )

    def _extract_state_features(
        self,
        sub_state: SubState,
        interpreter: NaturalLanguageInterpreter,
        sparse: bool = False,
    ) -> Dict[Text, List["Features"]]:
        # this method is called during both prediction and training,
        # `self._use_regex_interpreter == True` means that core was trained
        # separately, therefore substitute interpreter based on some trained
        # nlu model with default RegexInterpreter to make sure
        # that prediction and train time features are the same
        if self._use_regex_interpreter and not isinstance(
            interpreter, RegexInterpreter
        ):
            interpreter = RegexInterpreter()

        message = Message(data=sub_state)
        # remove entities from possible attributes
        attributes = set(
            attribute for attribute in sub_state.keys() if attribute != ENTITIES
        )

        parsed_message = interpreter.featurize_message(message)
        output = self._get_features_from_parsed_message(parsed_message, attributes)

        # check that name attributes have features
        name_attribute = self._get_name_attribute(attributes)
        if name_attribute and name_attribute not in output:
            # nlu pipeline didn't create features for user or action
            # this might happen, for example, when we have action_name in the state
            # but it did not get featurized because only character level
            # CountVectorsFeaturizer was included in the config.
            output[name_attribute] = self._create_features(
                sub_state, name_attribute, sparse
            )

        return output

    def encode_state(
        self, state: State, interpreter: NaturalLanguageInterpreter
    ) -> Dict[Text, List["Features"]]:
        """Encode the given state with the help of the given interpreter.

        Args:
            state: The state to encode
            interpreter: The interpreter used to encode the state

        Returns:
            A dictionary of state_type to list of features.
        """
        state_features = {}
        for state_type, sub_state in state.items():
            if state_type == PREVIOUS_ACTION:
                state_features.update(
                    self._extract_state_features(sub_state, interpreter, sparse=True)
                )
            # featurize user only if it is "real" user input,
            # i.e. input from a turn after action_listen
            if state_type == USER and is_prev_action_listen_in_state(state):
                state_features.update(
                    self._extract_state_features(sub_state, interpreter, sparse=True)
                )
                if sub_state.get(ENTITIES):
                    state_features[ENTITIES] = self._create_features(
                        sub_state, ENTITIES, sparse=True
                    )

            if state_type in {SLOTS, ACTIVE_LOOP}:
                state_features[state_type] = self._create_features(
                    sub_state, state_type, sparse=True
                )

        return state_features

    def encode_entities(
        self, entity_data: Dict[Text, Any], interpreter: NaturalLanguageInterpreter
    ) -> Dict[Text, List["Features"]]:
        """Encode the given entity data with the help of the given interpreter.

        Produce numeric entity tags for tokens.

        Args:
            entity_data: The dict containing the text and entity labels and locations
            interpreter: The interpreter used to encode the state

        Returns:
            A dictionary of entity type to list of features.
        """
        from rasa.nlu.test import determine_token_labels

        # TODO
        #  The entity states used to create the tag-idx-mapping contains the
        #  entities and the concatenated entity and roles/groups. We do not
        #  distinguish between entities and roles/groups right now.
        # TODO
        #  Should we support BILOU tagging?

        if TEXT not in entity_data or len(self.entity_tag_id_mapping) < 2:
            # we cannot build a classifier with fewer than 2 classes
            return {}

        parsed_text = interpreter.featurize_message(Message({TEXT: entity_data[TEXT]}))
        if not parsed_text:
            return {}
        entities = entity_data.get(ENTITIES, [])

        _tags = []
        for token in parsed_text.get(TOKENS_NAMES[TEXT], []):
            _tag = determine_token_labels(
                token, entities, attribute_key=ENTITY_ATTRIBUTE_TYPE
            )
            # TODO handle if tag is not in mapping
            _tags.append(self.entity_tag_id_mapping[_tag])

        # transpose to have seq_len x 1
        return {
            ENTITY_TAGS: [
                Features(np.array([_tags]).T, IDS, ENTITY_TAGS, TAG_ID_ORIGIN)
            ]
        }

    def _encode_action(
        self, action: Text, interpreter: NaturalLanguageInterpreter
    ) -> Dict[Text, List["Features"]]:
        if action in self.action_texts:
            action_as_sub_state = {ACTION_TEXT: action}
        else:
            action_as_sub_state = {ACTION_NAME: action}

        return self._extract_state_features(action_as_sub_state, interpreter)

    def encode_all_actions(
        self, domain: Domain, interpreter: NaturalLanguageInterpreter
    ) -> List[Dict[Text, List["Features"]]]:
        """Encode all action from the domain using the given interpreter.

        Args:
            domain: The domain that contains the actions.
            interpreter: The interpreter used to encode the actions.

        Returns:
            A list of encoded actions.
        """

        return [
            self._encode_action(action, interpreter)
            for action in domain.action_names_or_texts
        ]


class BinarySingleStateFeaturizer(SingleStateFeaturizer):
    def __init__(self) -> None:
        super().__init__()
        rasa.shared.utils.io.raise_warning(
            f"'{self.__class__.__name__}' is deprecated and "
            f"will be removed in the future. "
            f"It is recommended to use the '{SingleStateFeaturizer.__name__}' instead.",
            category=DeprecationWarning,
            docs=DOCS_URL_MIGRATION_GUIDE,
        )

    def _extract_state_features(
        self,
        sub_state: SubState,
        interpreter: NaturalLanguageInterpreter,
        sparse: bool = False,
    ) -> Dict[Text, List["Features"]]:
        # create a special method that doesn't use passed interpreter
        name_attribute = self._get_name_attribute(set(sub_state.keys()))
        if name_attribute:
            return {
                name_attribute: self._create_features(sub_state, name_attribute, sparse)
            }

        return {}


class LabelTokenizerSingleStateFeaturizer(SingleStateFeaturizer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # it is hard to fully mimic old behavior, but SingleStateFeaturizer
        # does the same thing if nlu pipeline is configured correctly
        rasa.shared.utils.io.raise_warning(
            f"'{self.__class__.__name__}' is deprecated and "
            f"will be removed in the future. "
            f"It is recommended to use the '{SingleStateFeaturizer.__name__}' instead.",
            category=DeprecationWarning,
            docs=DOCS_URL_MIGRATION_GUIDE,
        )
