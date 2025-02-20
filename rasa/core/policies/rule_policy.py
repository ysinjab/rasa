import logging
from typing import List, Dict, Text, Optional, Any, Set, TYPE_CHECKING, Tuple

from tqdm import tqdm
import numpy as np
import json
from collections import defaultdict

from rasa.shared.constants import DOCS_URL_RULES
from rasa.shared.exceptions import RasaException
import rasa.shared.utils.io
from rasa.shared.core.events import LoopInterrupted, UserUttered, ActionExecuted
from rasa.core.featurizers.tracker_featurizers import TrackerFeaturizer
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter
from rasa.core.policies.memoization import MemoizationPolicy
from rasa.core.policies.policy import SupportedData, PolicyPrediction
from rasa.shared.core.trackers import (
    DialogueStateTracker,
    get_active_loop_name,
    is_prev_action_listen_in_state,
)
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.core.constants import DEFAULT_CORE_FALLBACK_THRESHOLD, RULE_POLICY_PRIORITY
from rasa.shared.core.constants import (
    USER_INTENT_RESTART,
    USER_INTENT_BACK,
    USER_INTENT_SESSION_START,
    ACTION_LISTEN_NAME,
    ACTION_RESTART_NAME,
    ACTION_SESSION_START_NAME,
    ACTION_DEFAULT_FALLBACK_NAME,
    ACTION_BACK_NAME,
    RULE_SNIPPET_ACTION_NAME,
    SHOULD_NOT_BE_SET,
    PREVIOUS_ACTION,
    LOOP_REJECTED,
    LOOP_NAME,
    SLOTS,
    ACTIVE_LOOP,
)
from rasa.shared.core.domain import InvalidDomain, State, Domain
from rasa.shared.nlu.constants import ACTION_NAME, INTENT_NAME_KEY
import rasa.core.test
import rasa.core.training.training


if TYPE_CHECKING:
    from rasa.core.policies.ensemble import PolicyEnsemble

logger = logging.getLogger(__name__)

# These are Rasa Open Source default actions and overrule everything at any time.
DEFAULT_ACTION_MAPPINGS = {
    USER_INTENT_RESTART: ACTION_RESTART_NAME,
    USER_INTENT_BACK: ACTION_BACK_NAME,
    USER_INTENT_SESSION_START: ACTION_SESSION_START_NAME,
}

RULES = "rules"
RULES_FOR_LOOP_UNHAPPY_PATH = "rules_for_loop_unhappy_path"

LOOP_WAS_INTERRUPTED = "loop_was_interrupted"
DO_NOT_PREDICT_LOOP_ACTION = "do_not_predict_loop_action"

DEFAULT_RULES = "predicting default action"
LOOP_RULES = "handling active loops and forms"


class InvalidRule(RasaException):
    """Exception that can be raised when rules are not valid."""

    def __init__(self, message: Text) -> None:
        super().__init__()
        self.message = message

    def __str__(self) -> Text:
        return self.message + (
            f"\nYou can find more information about the usage of "
            f"rules at {DOCS_URL_RULES}. "
        )


class RulePolicy(MemoizationPolicy):
    """Policy which handles all the rules"""

    # rules use explicit json strings
    ENABLE_FEATURE_STRING_COMPRESSION = False

    # number of user inputs that is allowed in case rules are restricted
    ALLOWED_NUMBER_OF_USER_INPUTS = 1

    @staticmethod
    def supported_data() -> SupportedData:
        """The type of data supported by this policy.

        Returns:
            The data type supported by this policy (ML and rule data).
        """
        return SupportedData.ML_AND_RULE_DATA

    def __init__(
        self,
        featurizer: Optional[TrackerFeaturizer] = None,
        priority: int = RULE_POLICY_PRIORITY,
        lookup: Optional[Dict] = None,
        core_fallback_threshold: float = DEFAULT_CORE_FALLBACK_THRESHOLD,
        core_fallback_action_name: Text = ACTION_DEFAULT_FALLBACK_NAME,
        enable_fallback_prediction: bool = True,
        restrict_rules: bool = True,
        check_for_contradictions: bool = True,
        **kwargs: Any,
    ) -> None:
        """Create a `RulePolicy` object.

        Args:
            featurizer: `Featurizer` which is used to convert conversation states to
                features.
            priority: Priority of the policy which is used if multiple policies predict
                actions with the same confidence.
            lookup: Lookup table which is used to pick matching rules for a conversation
                state.
            core_fallback_threshold: Confidence of the prediction if no rule matched
                and de-facto threshold for a core fallback.
            core_fallback_action_name: Name of the action which should be predicted
                if no rule matched.
            enable_fallback_prediction: If `True` `core_fallback_action_name` is
                predicted in case no rule matched.
            restrict_rules: If `True` rules are restricted to contain a maximum of 1
                user message. This is used to avoid that users build a state machine
                using the rules.
            check_for_contradictions: Check for contradictions.
        """
        self._core_fallback_threshold = core_fallback_threshold
        self._fallback_action_name = core_fallback_action_name
        self._enable_fallback_prediction = enable_fallback_prediction
        self._restrict_rules = restrict_rules
        self._check_for_contradictions = check_for_contradictions

        self._prediction_source = None
        self._rules_sources = None

        # max history is set to `None` in order to capture any lengths of rule stories
        super().__init__(
            featurizer=featurizer,
            priority=priority,
            max_history=None,
            lookup=lookup,
            **kwargs,
        )

    @classmethod
    def validate_against_domain(
        cls, ensemble: Optional["PolicyEnsemble"], domain: Optional[Domain]
    ) -> None:
        if ensemble is None:
            return

        rule_policy = next(
            (p for p in ensemble.policies if isinstance(p, RulePolicy)), None
        )
        if not rule_policy or not rule_policy._enable_fallback_prediction:
            return

        if (
            domain is None
            or rule_policy._fallback_action_name not in domain.action_names
        ):
            raise InvalidDomain(
                f"The fallback action '{rule_policy._fallback_action_name}' which was "
                f"configured for the {RulePolicy.__name__} must be present in the "
                f"domain."
            )

    @staticmethod
    def _is_rule_snippet_state(state: State) -> bool:
        prev_action_name = state.get(PREVIOUS_ACTION, {}).get(ACTION_NAME)
        return prev_action_name == RULE_SNIPPET_ACTION_NAME

    def _create_feature_key(self, states: List[State]) -> Optional[Text]:
        new_states = []
        for state in reversed(states):
            if self._is_rule_snippet_state(state):
                # remove all states before RULE_SNIPPET_ACTION_NAME
                break
            new_states.insert(0, state)

        if not new_states:
            return

        # we sort keys to make sure that the same states
        # represented as dictionaries have the same json strings
        return json.dumps(new_states, sort_keys=True)

    @staticmethod
    def _states_for_unhappy_loop_predictions(states: List[State]) -> List[State]:
        """Modifies the states to create feature keys for loop unhappy path conditions.

        Args:
            states: a representation of a tracker
                as a list of dictionaries containing features

        Returns:
            modified states
        """
        # leave only last 2 dialogue turns to
        # - capture previous meaningful action before action_listen
        # - ignore previous intent
        if len(states) == 1 or not states[-2].get(PREVIOUS_ACTION):
            return [states[-1]]
        else:
            return [{PREVIOUS_ACTION: states[-2][PREVIOUS_ACTION]}, states[-1]]

    @staticmethod
    def _remove_rule_snippet_predictions(lookup: Dict[Text, Text]) -> Dict[Text, Text]:
        # Delete rules if it would predict the RULE_SNIPPET_ACTION_NAME action
        return {
            feature_key: action
            for feature_key, action in lookup.items()
            if action != RULE_SNIPPET_ACTION_NAME
        }

    def _create_loop_unhappy_lookup_from_states(
        self,
        trackers_as_states: List[List[State]],
        trackers_as_actions: List[List[Text]],
    ) -> Dict[Text, Text]:
        """Creates lookup dictionary from the tracker represented as states.

        Args:
            trackers_as_states: representation of the trackers as a list of states
            trackers_as_actions: representation of the trackers as a list of actions

        Returns:
            lookup dictionary
        """
        lookup = {}
        for states, actions in zip(trackers_as_states, trackers_as_actions):
            action = actions[0]
            active_loop = get_active_loop_name(states[-1])
            # even if there are two identical feature keys
            # their loop will be the same
            if not active_loop:
                continue

            states = self._states_for_unhappy_loop_predictions(states)
            feature_key = self._create_feature_key(states)
            if not feature_key:
                continue

            # Since rule snippets and stories inside the loop contain
            # only unhappy paths, notify the loop that
            # it was predicted after an answer to a different question and
            # therefore it should not validate user input
            if (
                # loop is predicted after action_listen in unhappy path,
                # therefore no validation is needed
                is_prev_action_listen_in_state(states[-1])
                and action == active_loop
            ):
                lookup[feature_key] = LOOP_WAS_INTERRUPTED
            elif (
                # some action other than active_loop is predicted in unhappy path,
                # therefore active_loop shouldn't be predicted by the rule
                not is_prev_action_listen_in_state(states[-1])
                and action != active_loop
            ):
                lookup[feature_key] = DO_NOT_PREDICT_LOOP_ACTION
        return lookup

    def _check_rule_restriction(
        self, rule_trackers: List[TrackerWithCachedStates]
    ) -> None:
        rules_exceeding_max_user_turns = []
        for tracker in rule_trackers:
            number_of_user_uttered = sum(
                isinstance(event, UserUttered) for event in tracker.events
            )
            if number_of_user_uttered > self.ALLOWED_NUMBER_OF_USER_INPUTS:
                rules_exceeding_max_user_turns.append(tracker.sender_id)

        if rules_exceeding_max_user_turns:
            raise InvalidRule(
                f"Found rules '{', '.join(rules_exceeding_max_user_turns)}' "
                f"that contain more than {self.ALLOWED_NUMBER_OF_USER_INPUTS} "
                f"user message. Rules are not meant to hardcode a state machine. "
                f"Please use stories for these cases."
            )

    @staticmethod
    def _expected_but_missing_slots(
        fingerprint: Dict[Text, List[Text]], state: State
    ) -> Set[Text]:
        expected_slots = set(fingerprint.get(SLOTS, {}))
        current_slots = set(state.get(SLOTS, {}).keys())
        # report all slots that are expected but aren't set in current slots
        return expected_slots.difference(current_slots)

    @staticmethod
    def _check_active_loops_fingerprint(
        fingerprint: Dict[Text, List[Text]], state: State
    ) -> Set[Text]:
        expected_active_loops = set(fingerprint.get(ACTIVE_LOOP, {}))
        # we don't use tracker.active_loop_name
        # because we need to keep should_not_be_set
        current_active_loop = state.get(ACTIVE_LOOP, {}).get(LOOP_NAME)
        if current_active_loop in expected_active_loops:
            # one of expected active loops is set
            return set()

        return expected_active_loops

    @staticmethod
    def _error_messages_from_fingerprints(
        action_name: Text,
        missing_fingerprint_slots: Set[Text],
        fingerprint_active_loops: Set[Text],
        rule_name: Text,
    ) -> List[Text]:
        error_messages = []
        if action_name and missing_fingerprint_slots:
            error_messages.append(
                f"- the action '{action_name}' in rule '{rule_name}' does not set some "
                f"of the slots that it sets in other rules. Slots not set in rule "
                f"'{rule_name}': '{', '.join(missing_fingerprint_slots)}'. Please "
                f"update the rule with an appropriate slot or if it is the last action "
                f"add 'wait_for_user_input: false' after this action."
            )
        if action_name and fingerprint_active_loops:
            # substitute `SHOULD_NOT_BE_SET` with `null` so that users
            # know what to put in their rules
            fingerprint_active_loops = set(
                "null" if active_loop == SHOULD_NOT_BE_SET else active_loop
                for active_loop in fingerprint_active_loops
            )
            # add action_name to active loop so that users
            # know what to put in their rules
            fingerprint_active_loops.add(action_name)

            error_messages.append(
                f"- the form '{action_name}' in rule '{rule_name}' does not set "
                f"the 'active_loop', that it sets in other rules: "
                f"'{', '.join(fingerprint_active_loops)}'. Please update the rule with "
                f"the appropriate 'active loop' property or if it is the last action "
                f"add 'wait_for_user_input: false' after this action."
            )
        return error_messages

    def _check_for_incomplete_rules(
        self, rule_trackers: List[TrackerWithCachedStates], domain: Domain
    ) -> None:
        logger.debug("Started checking if some rules are incomplete.")
        # we need to use only fingerprints from rules
        rule_fingerprints = rasa.core.training.training.create_action_fingerprints(
            rule_trackers, domain
        )
        if not rule_fingerprints:
            return

        error_messages = []
        for tracker in rule_trackers:
            states = tracker.past_states(domain)
            # the last action is always action listen
            action_names = [
                state.get(PREVIOUS_ACTION, {}).get(ACTION_NAME) for state in states[1:]
            ] + [ACTION_LISTEN_NAME]

            for state, action_name in zip(states, action_names):
                previous_action_name = state.get(PREVIOUS_ACTION, {}).get(ACTION_NAME)
                fingerprint = rule_fingerprints.get(previous_action_name)
                if (
                    not previous_action_name
                    or not fingerprint
                    or action_name == RULE_SNIPPET_ACTION_NAME
                    or previous_action_name == RULE_SNIPPET_ACTION_NAME
                ):
                    # do not check fingerprints for rule snippet action
                    # and don't raise if fingerprints are not satisfied
                    # for a previous action if current action is rule snippet action
                    continue

                missing_expected_slots = self._expected_but_missing_slots(
                    fingerprint, state
                )
                expected_active_loops = self._check_active_loops_fingerprint(
                    fingerprint, state
                )
                error_messages.extend(
                    self._error_messages_from_fingerprints(
                        previous_action_name,
                        missing_expected_slots,
                        expected_active_loops,
                        tracker.sender_id,
                    )
                )

        if error_messages:
            error_messages = "\n".join(error_messages)
            raise InvalidRule(
                f"\nIncomplete rules found🚨\n\n{error_messages}\n"
                f"Please note that if some slots or active loops should not be set "
                f"during prediction you need to explicitly set them to 'null' in the "
                f"rules."
            )

        logger.debug("Found no incompletions in rules.")

    def _predict_next_action(
        self,
        tracker: TrackerWithCachedStates,
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
    ) -> Optional[Text]:
        probabilities = self.predict_action_probabilities(
            tracker, domain, interpreter
        ).probabilities
        # do not raise an error if RulePolicy didn't predict anything for stories;
        # however for rules RulePolicy should always predict an action
        predicted_action_name = None
        if (
            probabilities != self._default_predictions(domain)
            or tracker.is_rule_tracker
        ):
            predicted_action_name = domain.action_names[np.argmax(probabilities)]

        return predicted_action_name

    def _check_prediction(
        self,
        tracker: TrackerWithCachedStates,
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        gold_action_name: Text,
        collect_sources: bool,
    ) -> Optional[Text]:

        predicted_action_name = self._predict_next_action(tracker, domain, interpreter)
        # if there is an active_loop,
        # RulePolicy will always predict active_loop first,
        # but inside loop unhappy path there might be another action
        if (
            tracker.active_loop_name
            and predicted_action_name != gold_action_name
            and predicted_action_name == tracker.active_loop_name
        ):
            rasa.core.test.emulate_loop_rejection(tracker)
            predicted_action_name = self._predict_next_action(
                tracker, domain, interpreter
            )

        if collect_sources:
            # we need to remember which action should be predicted by the rule
            # in order to correctly output the names of the contradicting rules
            rule_name = tracker.sender_id
            if self._prediction_source in {DEFAULT_RULES, LOOP_RULES}:
                # the real gold action contradict the one in the rules in this case
                gold_action_name = predicted_action_name
                rule_name = self._prediction_source

            self._rules_sources[self._prediction_source].append(
                (rule_name, gold_action_name)
            )
            return

        if not predicted_action_name or predicted_action_name == gold_action_name:
            return

        tracker_type = "rule" if tracker.is_rule_tracker else "story"
        contradicting_rules = {
            rule_name
            for rule_name, action_name in self._rules_sources[self._prediction_source]
            if action_name != gold_action_name
        }

        error_message = (
            f"- the prediction of the action '{gold_action_name}' in {tracker_type} "
            f"'{tracker.sender_id}' "
            f"is contradicting with rule(s) '{', '.join(contradicting_rules)}'"
        )
        # outputting predicted action 'action_default_fallback' is confusing
        if predicted_action_name != self._fallback_action_name:
            error_message += f" which predicted action '{predicted_action_name}'"

        return error_message + "."

    def _run_prediction_on_trackers(
        self,
        trackers: List[TrackerWithCachedStates],
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        collect_sources: bool,
    ) -> List[Text]:
        if collect_sources:
            self._rules_sources = defaultdict(list)

        error_messages = []
        pbar = tqdm(
            trackers,
            desc="Processed trackers",
            disable=rasa.shared.utils.io.is_logging_disabled(),
        )
        for tracker in pbar:
            running_tracker = tracker.init_copy()
            running_tracker.sender_id = tracker.sender_id
            # the first action is always unpredictable
            next_action_is_unpredictable = True
            for event in tracker.applied_events():
                if not isinstance(event, ActionExecuted):
                    running_tracker.update(event)
                    continue

                if event.action_name == RULE_SNIPPET_ACTION_NAME:
                    # notify that the action after RULE_SNIPPET_ACTION_NAME is
                    # unpredictable
                    next_action_is_unpredictable = True
                    running_tracker.update(event)
                    continue

                # do not run prediction on unpredictable actions
                if next_action_is_unpredictable or event.unpredictable:
                    next_action_is_unpredictable = False  # reset unpredictability
                    running_tracker.update(event)
                    continue

                gold_action_name = event.action_name or event.action_text
                error_message = self._check_prediction(
                    running_tracker,
                    domain,
                    interpreter,
                    gold_action_name,
                    collect_sources,
                )
                if error_message:
                    error_messages.append(error_message)

                running_tracker.update(event)

        return error_messages

    def _find_contradicting_rules(
        self,
        rule_trackers: List[TrackerWithCachedStates],
        all_trackers: List[TrackerWithCachedStates],
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
    ) -> None:
        logger.debug("Started checking rules and stories for contradictions.")
        # during training we run `predict_action_probabilities` to check for
        # contradicting rules.
        # We silent prediction debug to avoid too many logs during these checks.
        logger_level = logger.level
        logger.setLevel(logging.WARNING)

        # we need to run prediction on rule trackers twice, because we need to collect
        # information which rule snippets contributed to the learned rules
        self._run_prediction_on_trackers(
            rule_trackers, domain, interpreter, collect_sources=True
        )
        error_messages = self._run_prediction_on_trackers(
            all_trackers, domain, interpreter, collect_sources=False
        )

        logger.setLevel(logger_level)  # reset logger level
        if error_messages:
            error_messages = "\n".join(error_messages)
            raise InvalidRule(
                f"\nContradicting rules or stories found 🚨\n\n{error_messages}\n"
                f"Please update your stories and rules so that they don't contradict "
                f"each other."
            )

        logger.debug("Found no contradicting rules.")

    def train(
        self,
        training_trackers: List[TrackerWithCachedStates],
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> None:

        # only consider original trackers (no augmented ones)
        training_trackers = [
            t
            for t in training_trackers
            if not hasattr(t, "is_augmented") or not t.is_augmented
        ]
        # only use trackers from rule-based training data
        rule_trackers = [t for t in training_trackers if t.is_rule_tracker]
        if self._restrict_rules:
            self._check_rule_restriction(rule_trackers)

        if self._check_for_contradictions:
            self._check_for_incomplete_rules(rule_trackers, domain)

        (
            rule_trackers_as_states,
            rule_trackers_as_actions,
        ) = self.featurizer.training_states_and_actions(rule_trackers, domain)

        rules_lookup = self._create_lookup_from_states(
            rule_trackers_as_states, rule_trackers_as_actions
        )
        self.lookup[RULES] = self._remove_rule_snippet_predictions(rules_lookup)

        story_trackers = [t for t in training_trackers if not t.is_rule_tracker]
        (
            story_trackers_as_states,
            story_trackers_as_actions,
        ) = self.featurizer.training_states_and_actions(story_trackers, domain)

        # use all trackers to find negative rules in unhappy paths
        trackers_as_states = rule_trackers_as_states + story_trackers_as_states
        trackers_as_actions = rule_trackers_as_actions + story_trackers_as_actions

        # negative rules are not anti-rules, they are auxiliary to actual rules
        self.lookup[
            RULES_FOR_LOOP_UNHAPPY_PATH
        ] = self._create_loop_unhappy_lookup_from_states(
            trackers_as_states, trackers_as_actions
        )

        # make this configurable because checking might take a lot of time
        if self._check_for_contradictions:
            # using trackers here might not be the most efficient way, however
            # it allows us to directly test `predict_action_probabilities` method
            self._find_contradicting_rules(
                rule_trackers, training_trackers, domain, interpreter
            )

        logger.debug(f"Memorized '{len(self.lookup[RULES])}' unique rules.")

    @staticmethod
    def _does_rule_match_state(rule_state: State, conversation_state: State) -> bool:
        for state_type, rule_sub_state in rule_state.items():
            conversation_sub_state = conversation_state.get(state_type, {})
            for key, value in rule_sub_state.items():
                if isinstance(value, list):
                    # json dumps and loads tuples as lists,
                    # so we need to convert them back
                    value = tuple(value)

                if (
                    # value should be set, therefore
                    # check whether it is the same as in the state
                    value
                    and value != SHOULD_NOT_BE_SET
                    and conversation_sub_state.get(key) != value
                ) or (
                    # value shouldn't be set, therefore
                    # it should be None or non existent in the state
                    value == SHOULD_NOT_BE_SET
                    and conversation_sub_state.get(key)
                    # during training `SHOULD_NOT_BE_SET` is provided. Hence, we also
                    # have to check for the value of the slot state
                    and conversation_sub_state.get(key) != SHOULD_NOT_BE_SET
                ):
                    return False

        return True

    @staticmethod
    def _rule_key_to_state(rule_key: Text) -> List[State]:
        return json.loads(rule_key)

    def _is_rule_applicable(
        self, rule_key: Text, turn_index: int, conversation_state: State
    ) -> bool:
        """Check if rule is satisfied with current state at turn.

        Args:
            rule_key: the textual representation of learned rule
            turn_index: index of a current dialogue turn
            conversation_state: the state that corresponds to turn_index

        Returns:
            a boolean that says whether the rule is applicable to current state
        """
        # turn_index goes back in time
        reversed_rule_states = list(reversed(self._rule_key_to_state(rule_key)))

        # the rule must be applicable because we got (without any applicability issues)
        # further in the conversation history than the rule's length
        if turn_index >= len(reversed_rule_states):
            return True

        # a state has previous action if and only if it is not a conversation start
        # state
        current_previous_action = conversation_state.get(PREVIOUS_ACTION)
        rule_previous_action = reversed_rule_states[turn_index].get(PREVIOUS_ACTION)

        # current conversation state and rule state are conversation starters.
        # any slots with initial_value set will necessarily be in both states and don't
        # need to be checked.
        if not rule_previous_action and not current_previous_action:
            return True

        # current rule state is a conversation starter (due to conversation_start: true)
        # but current conversation state is not.
        # or
        # current conversation state is a starter
        # but current rule state is not.
        if not rule_previous_action or not current_previous_action:
            return False

        # check: current rule state features are present in current conversation state
        return self._does_rule_match_state(
            reversed_rule_states[turn_index], conversation_state
        )

    def _get_possible_keys(
        self, lookup: Dict[Text, Text], states: List[State]
    ) -> Set[Text]:
        possible_keys = set(lookup.keys())
        for i, state in enumerate(reversed(states)):
            # find rule keys that correspond to current state
            possible_keys = set(
                filter(
                    lambda _key: self._is_rule_applicable(_key, i, state), possible_keys
                )
            )
        return possible_keys

    @staticmethod
    def _find_action_from_default_actions(
        tracker: DialogueStateTracker,
    ) -> Optional[Text]:
        if (
            not tracker.latest_action_name == ACTION_LISTEN_NAME
            or not tracker.latest_message
        ):
            return None

        default_action_name = DEFAULT_ACTION_MAPPINGS.get(
            tracker.latest_message.intent.get(INTENT_NAME_KEY)
        )

        if default_action_name:
            logger.debug(f"Predicted default action '{default_action_name}'.")

        return default_action_name

    @staticmethod
    def _find_action_from_loop_happy_path(
        tracker: DialogueStateTracker,
    ) -> Optional[Text]:

        active_loop_name = tracker.active_loop_name
        active_loop_rejected = tracker.active_loop.get(LOOP_REJECTED)
        should_predict_loop = (
            active_loop_name
            and not active_loop_rejected
            and tracker.latest_action.get(ACTION_NAME) != active_loop_name
        )
        should_predict_listen = (
            active_loop_name
            and not active_loop_rejected
            and tracker.latest_action_name == active_loop_name
        )

        if should_predict_loop:
            logger.debug(f"Predicted loop '{active_loop_name}'.")
            return active_loop_name

        # predict `action_listen` if loop action was run successfully
        if should_predict_listen:
            logger.debug(
                f"Predicted '{ACTION_LISTEN_NAME}' after loop '{active_loop_name}'."
            )
            return ACTION_LISTEN_NAME

    def _find_action_from_rules(
        self, tracker: DialogueStateTracker, domain: Domain
    ) -> Tuple[Optional[Text], Optional[Text], bool]:
        """Predicts the next action based on the memoized rules.

        Args:
            tracker: The current conversation tracker.
            domain: The domain of the current model.

        Returns:
            A tuple of the predicted action name (or `None` if no matching rule was
            found), a description of the matching rule, and `True` if a loop action
            was predicted after the loop has been in an unhappy path before.
        """
        tracker_as_states = self.featurizer.prediction_states([tracker], domain)
        states = tracker_as_states[0]
        current_states = self.format_tracker_states(states)

        logger.debug(f"Current tracker state:{current_states}")

        # Tracks if we are returning after an unhappy loop path. If this becomes `True`
        # the policy returns an event which notifies the loop action that it
        # is returning after an unhappy path. For example, the `FormAction` uses this
        # to skip the validation of slots for its first execution after an unhappy path.
        returning_from_unhappy_path = False

        rule_keys = self._get_possible_keys(self.lookup[RULES], states)
        predicted_action_name = None
        best_rule_key = ""
        if rule_keys:
            # if there are several rules,
            # it should mean that some rule is a subset of another rule
            # therefore we pick a rule of maximum length
            best_rule_key = max(rule_keys, key=len)
            predicted_action_name = self.lookup[RULES].get(best_rule_key)

        active_loop_name = tracker.active_loop_name
        if active_loop_name:
            # find rules for unhappy path of the loop
            loop_unhappy_keys = self._get_possible_keys(
                self.lookup[RULES_FOR_LOOP_UNHAPPY_PATH], states
            )
            # there could be several unhappy path conditions
            unhappy_path_conditions = [
                self.lookup[RULES_FOR_LOOP_UNHAPPY_PATH].get(key)
                for key in loop_unhappy_keys
            ]

            # Check if a rule that predicted action_listen
            # was applied inside the loop.
            # Rules might not explicitly switch back to the loop.
            # Hence, we have to take care of that.
            predicted_listen_from_general_rule = (
                predicted_action_name == ACTION_LISTEN_NAME
                and not get_active_loop_name(self._rule_key_to_state(best_rule_key)[-1])
            )
            if predicted_listen_from_general_rule:
                if DO_NOT_PREDICT_LOOP_ACTION not in unhappy_path_conditions:
                    # negative rules don't contain a key that corresponds to
                    # the fact that active_loop shouldn't be predicted
                    logger.debug(
                        f"Predicted loop '{active_loop_name}' by overwriting "
                        f"'{ACTION_LISTEN_NAME}' predicted by general rule."
                    )
                    return active_loop_name, LOOP_RULES, returning_from_unhappy_path

                # do not predict anything
                predicted_action_name = None

            if LOOP_WAS_INTERRUPTED in unhappy_path_conditions:
                logger.debug(
                    "Returning from unhappy path. Loop will be notified that "
                    "it was interrupted."
                )
                returning_from_unhappy_path = True

        if predicted_action_name is not None:
            logger.debug(
                f"There is a rule for the next action '{predicted_action_name}'."
            )
        else:
            logger.debug("There is no applicable rule.")

        # if we didn't predict anything from the rules, then the feature key created
        # from states can be used as an indicator that this state will lead to fallback
        return (
            predicted_action_name,
            best_rule_key or self._create_feature_key(states),
            returning_from_unhappy_path,
        )

    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> PolicyPrediction:
        """Predicts the next action (see parent class for more information)."""
        result = self._default_predictions(domain)

        # Rasa Open Source default actions overrule anything. If users want to achieve
        # the same, they need to write a rule or make sure that their loop rejects
        # accordingly.
        default_action_name = self._find_action_from_default_actions(tracker)
        if default_action_name:
            self._prediction_source = DEFAULT_RULES
            return self._prediction(
                self._prediction_result(default_action_name, tracker, domain)
            )

        # A loop has priority over any other rule.
        # The rules or any other prediction will be applied only if a loop was rejected.
        # If we are in a loop, and the loop didn't run previously or rejected, we can
        # simply force predict the loop.
        loop_happy_path_action_name = self._find_action_from_loop_happy_path(tracker)
        if loop_happy_path_action_name:
            self._prediction_source = LOOP_RULES
            return self._prediction(
                self._prediction_result(loop_happy_path_action_name, tracker, domain)
            )

        (
            rules_action_name,
            source,
            returning_from_unhappy_path,
        ) = self._find_action_from_rules(tracker, domain)
        # we want to remember the source even if rules didn't predict any action
        self._prediction_source = source

        policy_events = [LoopInterrupted(True)] if returning_from_unhappy_path else []

        if rules_action_name:
            result = self._prediction_result(rules_action_name, tracker, domain)

        return self._prediction(result, events=policy_events)

    def _default_predictions(self, domain: Domain) -> List[float]:
        result = super()._default_predictions(domain)

        if self._enable_fallback_prediction:
            result[
                domain.index_for_action(self._fallback_action_name)
            ] = self._core_fallback_threshold
        return result

    def _metadata(self) -> Dict[Text, Any]:
        return {
            "priority": self.priority,
            "lookup": self.lookup,
            "core_fallback_threshold": self._core_fallback_threshold,
            "core_fallback_action_name": self._fallback_action_name,
            "enable_fallback_prediction": self._enable_fallback_prediction,
        }

    @classmethod
    def _metadata_filename(cls) -> Text:
        return "rule_policy.json"
