#!/usr/bin/env python3
"""Test script to reproduce the UnboundLocalError bug in Django's deserialize_m2m_values."""

import sys
import traceback

# Add Django to the path
sys.path.insert(0, '/home/npc/miniconda/lib/python3.13/site-packages')

from django.core.serializers.base import deserialize_m2m_values, M2MDeserializationError


class ErrorIterator:
    """Iterator that raises an exception on first __next__ call."""
    def __iter__(self):
        return self

    def __next__(self):
        raise RuntimeError("Error during iteration")


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


def test_unboundlocal_error():
    """Test that demonstrates the UnboundLocalError."""
    field = MockField()
    field_value = ErrorIterator()

    print("Testing deserialize_m2m_values with ErrorIterator that raises immediately...")
    try:
        result = deserialize_m2m_values(field, field_value, using='default', handle_forward_references=False)
        print(f"Unexpected success: {result}")
    except UnboundLocalError as e:
        print(f"✓ UnboundLocalError raised as expected: {e}")
        print("This is the bug - should have raised M2MDeserializationError instead!")
        traceback.print_exc()
        return True
    except M2MDeserializationError as e:
        print(f"✗ M2MDeserializationError raised (bug might be fixed): {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected exception: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def test_normal_operation():
    """Test that normal operation works correctly."""
    field = MockField()
    field_value = [1, 2, 3]

    print("\nTesting normal operation with list [1, 2, 3]...")
    try:
        result = deserialize_m2m_values(field, field_value, using='default', handle_forward_references=False)
        print(f"✓ Success: {result}")
        return True
    except Exception as e:
        print(f"✗ Unexpected exception in normal case: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Reproducing Django deserialize_m2m_values UnboundLocalError bug")
    print("=" * 60)

    bug_exists = test_unboundlocal_error()
    normal_works = test_normal_operation()

    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Bug exists (UnboundLocalError raised): {bug_exists}")
    print(f"Normal operation works: {normal_works}")
    print("=" * 60)