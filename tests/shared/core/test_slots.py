from typing import Any, Tuple, List

import pytest
from _pytest.fixtures import SubRequest

import rasa.shared.core.constants
from rasa.shared.core.slots import (
    InvalidSlotTypeException,
    Slot,
    TextSlot,
    BooleanSlot,
    FloatSlot,
    ListSlot,
    UnfeaturizedSlot,
    CategoricalSlot,
    bool_from_any,
    AnySlot,
)


class SlotTestCollection:
    """Tests every slot needs to fulfill.

    Each slot can declare further tests on its own."""

    def create_slot(self, influence_conversation: bool) -> Slot:
        raise NotImplementedError

    def value_feature_pair(self, request: SubRequest) -> Tuple[Any, List[float]]:
        """Values where featurization is defined and should be tested."""
        raise NotImplementedError

    def invalid_value(self, request: SubRequest) -> Any:
        """Values, that should be handled gracefully but where the
        featurization is not defined."""
        raise NotImplementedError

    def test_featurization(self, value_feature_pair: Tuple[Any, List[float]]):
        slot = self.create_slot(influence_conversation=True)
        value, expected = value_feature_pair
        slot.value = value
        assert slot.as_feature() == expected
        assert (
            len(slot.as_feature()) == slot.feature_dimensionality()
        ), "Wrong feature dimensionality"

        # now reset the slot to get initial value again
        slot.reset()
        assert (
            slot.value == slot.initial_value
        ), "Slot should be reset to its initial value"

    def test_empty_slot_featurization(self):
        slot = self.create_slot(influence_conversation=True)
        assert (
            slot.value == slot.initial_value
        ), "An empty slot should be set to the initial value"
        assert len(slot.as_feature()) == slot.feature_dimensionality()

    def test_featurization_if_marked_as_unfeaturized(
        self, value_feature_pair: Tuple[Any, List[float]]
    ):
        slot = self.create_slot(influence_conversation=False)
        value, _ = value_feature_pair
        slot.value = value

        features = slot.as_feature()
        assert features == []

        dimensions = slot.feature_dimensionality()
        assert dimensions == 0

    def test_has_a_type_name(self):
        slot = self.create_slot(influence_conversation=True)
        assert slot.type_name is not None
        assert type(slot) == Slot.resolve_by_type(slot.type_name)

    def test_handles_invalid_values(self, invalid_value: Any):
        slot = self.create_slot(influence_conversation=True)
        slot.value = invalid_value
        assert slot.as_feature() is not None
        assert len(slot.as_feature()) == slot.feature_dimensionality()

    @pytest.mark.parametrize("influence_conversation", [True, False])
    def test_serialization(self, influence_conversation: bool):
        slot = self.create_slot(influence_conversation)

        persistence_info = slot.persistence_info()

        slot_type = Slot.resolve_by_type(persistence_info.get("type"))
        recreated = slot_type(
            slot.name, **{k: v for k, v in persistence_info.items() if k != "type"}
        )

        assert isinstance(slot, slot_type)
        assert recreated.persistence_info() == persistence_info


class TestTextSlot(SlotTestCollection):
    def create_slot(self, influence_conversation: bool) -> Slot:
        return TextSlot("test", influence_conversation=influence_conversation)

    @pytest.fixture(params=[1, {"a": "b"}, 2.0, [], True])
    def invalid_value(self, request: SubRequest) -> Any:
        return request.param

    @pytest.fixture(
        params=[
            (None, [0]),
            ("", [1]),
            ("some test string", [1]),
            ("some test string 🌴", [1]),
        ]
    )
    def value_feature_pair(self, request: SubRequest) -> Tuple[Any, List[float]]:
        return request.param


class TestBooleanSlot(SlotTestCollection):
    def create_slot(self, influence_conversation: bool) -> Slot:
        return BooleanSlot("test", influence_conversation=influence_conversation)

    @pytest.fixture(params=[{"a": "b"}, [], "asd", "🌴"])
    def invalid_value(self, request: SubRequest) -> Any:
        return request.param

    @pytest.fixture(
        params=[
            (None, [0, 0]),
            (True, [1, 1]),
            ("9", [1, 0]),
            (12, [1, 0]),
            (False, [1, 0]),
            ("0", [1, 0]),
            (0, [1, 0]),
            ("true", [1, 1]),
            ("True", [1, 1]),
            ("false", [1, 0]),
            ("False", [1, 0]),
        ]
    )
    def value_feature_pair(self, request: SubRequest) -> Tuple[Any, List[float]]:
        return request.param


def test_bool_from_any_raises_value_error():
    with pytest.raises(ValueError):
        bool_from_any("abc")


def test_bool_from_any_raises_type_error():
    with pytest.raises(TypeError):
        bool_from_any(None)


class TestFloatSlot(SlotTestCollection):
    def create_slot(self, influence_conversation: bool = False) -> Slot:
        return FloatSlot("test", influence_conversation=influence_conversation)

    @pytest.fixture(params=[{"a": "b"}, [], "asd", "🌴"])
    def invalid_value(self, request: SubRequest) -> Any:
        return request.param

    @pytest.fixture(
        params=[
            (None, [0, 0]),
            (True, [1, 1]),
            (2.0, [1, 1]),
            (1.0, [1, 1]),
            (0.5, [1, 0.5]),
            (0, [1, 0]),
            (-0.5, [1, 0.0]),
        ]
    )
    def value_feature_pair(self, request: SubRequest) -> Tuple[Any, List[float]]:
        return request.param


class TestListSlot(SlotTestCollection):
    def create_slot(self, influence_conversation: bool) -> Slot:
        return ListSlot("test", influence_conversation=influence_conversation)

    @pytest.fixture(params=[{"a": "b"}, 1, True, "asd", "🌴"])
    def invalid_value(self, request: SubRequest) -> Any:
        return request.param

    @pytest.fixture(params=[(None, [0]), ([], [0]), ([1], [1]), (["asd", 1, {}], [1])])
    def value_feature_pair(self, request: SubRequest) -> Tuple[Any, List[float]]:
        return request.param


class TestUnfeaturizedSlot(SlotTestCollection):
    def create_slot(self, influence_conversation: bool) -> Slot:
        return UnfeaturizedSlot("test", influence_conversation=False)

    @pytest.fixture(params=["there is nothing invalid, but we need to pass something"])
    def invalid_value(self, request: SubRequest) -> Any:
        return request.param

    @pytest.fixture(params=[(None, []), ([23], []), (1, []), ("asd", [])])
    def value_feature_pair(self, request: SubRequest) -> Tuple[Any, List[float]]:
        return request.param

    def test_exception_if_featurized(self):
        with pytest.raises(ValueError):
            UnfeaturizedSlot("⛔️", influence_conversation=True)

    def test_deprecation_warning(self):
        with pytest.warns(FutureWarning):
            self.create_slot(False)


class TestCategoricalSlot(SlotTestCollection):
    def create_slot(self, influence_conversation: bool) -> Slot:
        return CategoricalSlot(
            "test",
            values=[1, "two", "小于", {"three": 3}, None],
            influence_conversation=influence_conversation,
        )

    @pytest.fixture(params=[{"a": "b"}, 2, True, "asd", "🌴"])
    def invalid_value(self, request: SubRequest) -> Any:
        return request.param

    @pytest.fixture(
        params=[
            (None, [0, 0, 0, 0, 1]),
            (1, [1, 0, 0, 0, 0]),
            ("two", [0, 1, 0, 0, 0]),
            ("小于", [0, 0, 1, 0, 0]),
            ({"three": 3}, [0, 0, 0, 1, 0]),
            (
                rasa.shared.core.constants.DEFAULT_CATEGORICAL_SLOT_VALUE,
                [0, 0, 0, 0, 0],
            ),
        ]
    )
    def value_feature_pair(self, request: SubRequest) -> Tuple[Any, List[float]]:
        return request.param


class TestCategoricalSlotDefaultValue(SlotTestCollection):
    def create_slot(self, influence_conversation: bool) -> Slot:
        slot = CategoricalSlot(
            "test",
            values=[1, "two", "小于", {"three": 3}, None],
            influence_conversation=influence_conversation,
        )
        slot.add_default_value()
        return slot

    @pytest.fixture(params=[{"a": "b"}, 2, True, "asd", "🌴"])
    def invalid_value(self, request: SubRequest) -> Any:
        return request.param

    @pytest.fixture(
        params=[
            (None, [0, 0, 0, 0, 1, 0]),
            (1, [1, 0, 0, 0, 0, 0]),
            ("two", [0, 1, 0, 0, 0, 0]),
            ("小于", [0, 0, 1, 0, 0, 0]),
            ({"three": 3}, [0, 0, 0, 1, 0, 0]),
            (
                rasa.shared.core.constants.DEFAULT_CATEGORICAL_SLOT_VALUE,
                [0, 0, 0, 0, 0, 1],
            ),
            ("unseen value", [0, 0, 0, 0, 0, 1]),
        ]
    )
    def value_feature_pair(self, request: SubRequest) -> Tuple[Any, List[float]]:
        return request.param


class TestAnySlot(SlotTestCollection):
    def create_slot(self, influence_conversation: bool) -> Slot:
        return AnySlot("test", influence_conversation=False)

    @pytest.fixture(params=["there is nothing invalid, but we need to pass something"])
    def invalid_value(self, request: SubRequest) -> Any:
        return request.param

    @pytest.fixture(
        params=[
            (None, []),
            ([], []),
            ({"nested": {"dict": [1, 2, 3]}}, []),
            (["asd", 1, {}], []),
        ]
    )
    def value_feature_pair(self, request: SubRequest) -> Tuple[Any, List[float]]:
        return request.param

    def test_exception_if_featurized(self):
        with pytest.raises(ValueError):
            UnfeaturizedSlot("⛔️", influence_conversation=True)


def test_raises_on_invalid_slot_type():
    with pytest.raises(InvalidSlotTypeException):
        Slot.resolve_by_type("foobar")
