import numpy as np
from hypothesis import given, strategies as st

# First, let's test the exact reproduction case from the bug report
kind_mapping = {
    "bool": np.bool_,
    "signed integer": np.signedinteger,
    "unsigned integer": np.unsignedinteger,
    "integral": np.integer,
    "real floating": np.floating,
    "complex floating": np.complexfloating,
    "numeric": np.number,
}

def isdtype_fallback(dtype, kind):
    kinds = kind if isinstance(kind, tuple) else (kind,)
    str_kinds = {k for k in kinds if isinstance(k, str)}
    type_kinds = {k.type for k in kinds if isinstance(k, np.dtype)}

    if unknown_kind_types := set(kinds) - str_kinds - type_kinds:
        raise TypeError(
            f"kind must be str, np.dtype or a tuple of these, got {unknown_kind_types}"
        )

    if unknown_kinds := {k for k in str_kinds if k not in kind_mapping}:
        raise ValueError(
            f"unknown kind: {unknown_kinds}, must be a np.dtype or one of {list(kind_mapping)}"
        )

    translated_kinds = {kind_mapping[k] for k in str_kinds} | type_kinds
    if isinstance(dtype, np.generic):
        return isinstance(dtype, translated_kinds)
    else:
        return any(np.issubdtype(dtype, k) for k in translated_kinds)

print("Testing with list input: isdtype_fallback(np.int32, ['signed integer', 'bool'])")
try:
    result = isdtype_fallback(np.int32, ["signed integer", "bool"])
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")

print("\nTesting with tuple input (expected to work): isdtype_fallback(np.int32, ('signed integer', 'bool'))")
try:
    result = isdtype_fallback(np.int32, ("signed integer", "bool"))
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")

# Now let's trace through exactly what happens with a list
print("\n--- Tracing what happens with list input ---")
kind = ["signed integer", "bool"]
print(f"Input kind: {kind}")
print(f"isinstance(kind, tuple): {isinstance(kind, tuple)}")
kinds = kind if isinstance(kind, tuple) else (kind,)
print(f"After conversion, kinds = {kinds}")
print(f"Type of kinds: {type(kinds)}")
print(f"Contents of kinds tuple: {[type(x) for x in kinds]}")

# Try to create a set with kinds
print("\nTrying to create set(kinds)...")
try:
    s = set(kinds)
    print(f"Success: {s}")
except TypeError as e:
    print(f"TypeError: {e}")

# Test the hypothesis test from the bug report
print("\n--- Running hypothesis test ---")
@given(st.sampled_from([np.int32, np.float64, np.bool_]))
def test_isdtype_accepts_list_of_kinds(dtype):
    try:
        result = isdtype_fallback(dtype, ["signed integer", "bool"])
    except TypeError as e:
        assert "list" not in str(e).lower() or "unhashable" not in str(e).lower(), \
            f"Should give clear error about list not being supported, not: {e}"

# Run a few examples
for dtype in [np.int32, np.float64, np.bool_]:
    print(f"\nTesting dtype={dtype}")
    try:
        test_isdtype_accepts_list_of_kinds(dtype)
        print("Test passed (no assertion error)")
    except AssertionError as e:
        print(f"Assertion failed: {e}")