#!/usr/bin/env python3
"""Test to reproduce the pydantic.v1 hash/equality bug"""

from pydantic.v1 import BaseModel

# Test 1: Basic reproduction
print("Test 1: Basic reproduction")
print("-" * 40)

class Model1(BaseModel):
    class Config:
        frozen = True
    x: int

class Model2(BaseModel):
    class Config:
        frozen = True
    x: int

m1 = Model1(x=42)
m2 = Model2(x=42)

print(f"m1 = Model1(x=42): {m1}")
print(f"m2 = Model2(x=42): {m2}")
print(f"m1 == m2: {m1 == m2}")
print(f"hash(m1): {hash(m1)}")
print(f"hash(m2): {hash(m2)}")
print(f"hash(m1) == hash(m2): {hash(m1) == hash(m2)}")

# Test the Python contract violation
if m1 == m2:
    if hash(m1) != hash(m2):
        print("❌ VIOLATION: m1 == m2 but hash(m1) != hash(m2)")
    else:
        print("✓ OK: m1 == m2 and hash(m1) == hash(m2)")

print()

# Test 2: Set membership issue
print("Test 2: Set membership behavior")
print("-" * 40)
s = {m1}
print(f"Created set with m1: {s}")
print(f"m2 in s: {m2 in s}")
print(f"Expected: True (since m1 == m2)")
print()

# Test 3: Dict key issue
print("Test 3: Dict key behavior")
print("-" * 40)
d = {m1: "value1"}
print(f"Created dict with m1 as key")
try:
    value = d[m2]
    print(f"d[m2]: {value}")
except KeyError:
    print(f"KeyError: m2 not found in dict (even though m1 == m2)")

print()

# Test 4: Property-based test with Hypothesis
print("Test 4: Property-based test with Hypothesis")
print("-" * 40)
try:
    from hypothesis import given, strategies as st

    def make_model_class(name: str):
        class DynamicModel(BaseModel):
            class Config:
                frozen = True
            x: int
        DynamicModel.__name__ = name
        DynamicModel.__qualname__ = name
        return DynamicModel

    @given(st.integers(), st.text(min_size=1, max_size=10), st.text(min_size=1, max_size=10))
    def test_equal_objects_have_equal_hash(value, name1, name2):
        Model1 = make_model_class(name1)
        Model2 = make_model_class(name2)

        m1 = Model1(x=value)
        m2 = Model2(x=value)

        if m1 == m2:
            assert hash(m1) == hash(m2), \
                f"Equal objects must have equal hash: {m1} == {m2} but hash differs"

    # Run a limited test
    test_equal_objects_have_equal_hash()
    print("Hypothesis test failed as expected - hash/equality contract violated")
except AssertionError as e:
    print(f"Hypothesis test caught violation: {e}")
except Exception as e:
    print(f"Hypothesis test error: {e}")