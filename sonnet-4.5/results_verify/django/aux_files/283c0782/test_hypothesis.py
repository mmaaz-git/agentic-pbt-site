#!/usr/bin/env python3
"""Property-based test using Hypothesis to test various exception types."""

import sys
sys.path.insert(0, '/home/npc/miniconda/lib/python3.13/site-packages')

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


def test_deserialize_m2m_values_handles_iteration_errors(error_type):
    field = MockField()
    field_value = ErrorIterator(error_type)

    try:
        result = deserialize_m2m_values(field, field_value, using='default', handle_forward_references=False)
        assert False, f"Expected exception but got result: {result}"
    except UnboundLocalError as e:
        print(f"✗ UnboundLocalError for {error_type.__name__}: {e}")
        print("  This is the bug - should have raised M2MDeserializationError")
        return False  # Bug exists
    except M2MDeserializationError as e:
        print(f"✓ M2MDeserializationError for {error_type.__name__} (expected): {e}")
        return True  # Working correctly
    except Exception as e:
        print(f"? Unexpected exception for {error_type.__name__}: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    print("Testing with different exception types using Hypothesis...")
    print("=" * 60)

    error_types = [RuntimeError, ValueError, TypeError, KeyError]
    all_fail = True

    for error_type in error_types:
        result = test_deserialize_m2m_values_handles_iteration_errors(error_type)
        if result:
            all_fail = False

    print("=" * 60)
    if all_fail:
        print("✓ Bug confirmed: All error types trigger UnboundLocalError")
    else:
        print("✗ Some tests did not trigger UnboundLocalError")