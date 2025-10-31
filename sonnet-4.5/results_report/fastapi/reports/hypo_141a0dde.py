#!/usr/bin/env python3
"""Hypothesis test for circular dependency detection in FastAPI's get_flat_dependant"""

from hypothesis import given, strategies as st
from fastapi.dependencies.models import Dependant
from fastapi.dependencies.utils import get_flat_dependant


def test_circular_dependency_detection():
    """Test that get_flat_dependant handles circular dependencies"""
    dep1 = Dependant(call=lambda: "dep1", name="dep1")
    dep2 = Dependant(call=lambda: "dep2", name="dep2", dependencies=[dep1])
    dep1.dependencies.append(dep2)

    flat = get_flat_dependant(dep1)
    assert isinstance(flat, Dependant)


if __name__ == "__main__":
    try:
        test_circular_dependency_detection()
        print("Test passed!")
    except RecursionError as e:
        print("Test failed with RecursionError!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()