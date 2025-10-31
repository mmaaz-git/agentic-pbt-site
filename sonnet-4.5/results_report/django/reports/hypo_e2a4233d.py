from hypothesis import given, strategies as st, settings
from django.core.serializers.base import deserialize_m2m_values, M2MDeserializationError
import pytest


class ErrorIterator:
    def __init__(self, error_type):
        self.error_type = error_type

    def __iter__(self):
        return self

    def __next__(self):
        raise self.error_type("Error during iteration")


@given(st.sampled_from([RuntimeError, ValueError, TypeError, KeyError]))
@settings(max_examples=20)
def test_deserialize_m2m_values_handles_iteration_errors(error_type):
    class MockPK:
        def to_python(self, v):
            return int(v)

    class MockMeta:
        pk = MockPK()

    class MockDefaultManager:
        pass

    class MockModel:
        _meta = MockMeta()
        _default_manager = MockDefaultManager()

    class MockRemoteField:
        model = MockModel

    class MockField:
        remote_field = MockRemoteField()

    field = MockField()
    field_value = ErrorIterator(error_type)

    with pytest.raises((M2MDeserializationError, UnboundLocalError)) as exc_info:
        deserialize_m2m_values(field, field_value, using='default', handle_forward_references=False)

    if isinstance(exc_info.value, UnboundLocalError):
        pytest.fail(f"UnboundLocalError raised. This is a bug in deserialize_m2m_values.")


if __name__ == "__main__":
    test_deserialize_m2m_values_handles_iteration_errors()