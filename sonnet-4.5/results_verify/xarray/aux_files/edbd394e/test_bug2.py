import numpy as np

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

print("=== MAIN BUG REPRODUCTION ===")
print("\n1. Testing with list input: isdtype_fallback(np.int32, ['signed integer', 'bool'])")
try:
    result = isdtype_fallback(np.int32, ["signed integer", "bool"])
    print(f"   Result: {result}")
except Exception as e:
    print(f"   Exception raised: {type(e).__name__}: {e}")
    print(f"   Does error mention 'unhashable': {'unhashable' in str(e).lower()}")
    print(f"   Does error mention 'list': {'list' in str(e).lower()}")

print("\n2. Testing with tuple input (expected to work): isdtype_fallback(np.int32, ('signed integer', 'bool'))")
try:
    result = isdtype_fallback(np.int32, ("signed integer", "bool"))
    print(f"   Result: {result}")
except Exception as e:
    print(f"   Exception raised: {type(e).__name__}: {e}")

print("\n=== TRACING THE BUG ===")
kind = ["signed integer", "bool"]
print(f"Input kind: {kind}")
print(f"isinstance(kind, tuple): {isinstance(kind, tuple)}")
kinds = kind if isinstance(kind, tuple) else (kind,)
print(f"After conversion, kinds = {kinds}")
print(f"kinds is a tuple containing: {[f'{x} (type: {type(x).__name__})' for x in kinds]}")

print("\nWhen we try set(kinds):")
try:
    s = set(kinds)
    print(f"   Success: {s}")
except TypeError as e:
    print(f"   TypeError raised: {e}")
    print(f"   This happens because kinds contains a list object, and lists are unhashable")

print("\n=== TESTING NUMPY 2.0+ BEHAVIOR ===")
print("Checking if numpy.isdtype is available (numpy >= 2.0):")
try:
    from numpy import isdtype as np_isdtype
    print("   numpy.isdtype is available")
    print("\n   Testing numpy.isdtype with list:")
    try:
        result = np_isdtype(np.int32, ["signed integer", "bool"])
        print(f"   Result: {result}")
    except Exception as e:
        print(f"   Exception: {type(e).__name__}: {e}")

    print("\n   Testing numpy.isdtype with tuple:")
    try:
        result = np_isdtype(np.int32, ("signed integer", "bool"))
        print(f"   Result: {result}")
    except Exception as e:
        print(f"   Exception: {type(e).__name__}: {e}")
except ImportError:
    print("   numpy.isdtype is NOT available (numpy < 2.0)")

print("\n=== ANALYSIS ===")
print("The bug report claims:")
print("1. The error 'unhashable type: list' is confusing")
print("2. Lists are a 'natural input type' and users might reasonably pass them")
print("3. The numpy >= 2.0 implementation 'properly rejects lists with a clear error'")
print("\nActual findings:")
print("- The function signature explicitly specifies 'tuple[DTypeLike, ...]' not list")
print("- The documentation (both numpy and xarray) never mentions lists as acceptable")
print("- The error occurs because the code wraps a list in a tuple: (list,) instead of converting it")