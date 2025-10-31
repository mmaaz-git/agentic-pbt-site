import numpy as np
from hypothesis import given, strategies as st, settings
from scipy.spatial.distance import jensenshannon
import warnings
import sys

print("=" * 60)
print("Testing Jensen-Shannon Distance with Invalid Base Parameters")
print("=" * 60)

# First, test the Hypothesis property test
print("\n1. Running Hypothesis property test:")
print("-" * 40)

failed_examples = []

@settings(max_examples=500)
@given(
    st.integers(min_value=2, max_value=10),
    st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)
)
def test_jensenshannon_base_produces_finite_result(k, base):
    x = np.random.rand(k) + 0.1
    y = np.random.rand(k) + 0.1
    x = x / x.sum()
    y = y / y.sum()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = jensenshannon(x, y, base=base)

        if not np.isfinite(result):
            failed_examples.append({
                'base': base,
                'result': result,
                'warnings': [str(warning.message) for warning in w]
            })

# Run the hypothesis test
try:
    test_jensenshannon_base_produces_finite_result()
    print("Hypothesis test completed.")
except Exception as e:
    print(f"Hypothesis test failed with error: {e}")

# Show some failed examples
if failed_examples:
    print(f"\nFound {len(failed_examples)} examples that produce non-finite results:")
    for i, example in enumerate(failed_examples[:5]):  # Show first 5
        print(f"  Example {i+1}: base={example['base']:.3f}, result={example['result']}")
        if example['warnings']:
            print(f"    Warnings: {example['warnings'][0]}")

# Test the specific case with base=1.0
print("\n2. Testing specific case with base=1.0:")
print("-" * 40)

p = np.array([0.5, 0.5])
q = np.array([0.3, 0.7])

print(f"p = {p}")
print(f"q = {q}")

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result = jensenshannon(p, q, base=1.0)

    print(f"\nResult: {result}")
    print(f"Is infinite: {np.isinf(result)}")
    print(f"Is NaN: {np.isnan(result)}")

    if w:
        print(f"\nWarnings raised ({len(w)}):")
        for warning in w:
            print(f"  - {warning.category.__name__}: {warning.message}")
    else:
        print("\nNo warnings raised")

# Test with base=0
print("\n3. Testing with base=0:")
print("-" * 40)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result = jensenshannon(p, q, base=0)

    print(f"Result: {result}")
    print(f"Is infinite: {np.isinf(result)}")
    print(f"Is NaN: {np.isnan(result)}")

    if w:
        print(f"\nWarnings raised ({len(w)}):")
        for warning in w:
            print(f"  - {warning.category.__name__}: {warning.message}")

# Test with negative base
print("\n4. Testing with base=-1:")
print("-" * 40)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result = jensenshannon(p, q, base=-1)

    print(f"Result: {result}")
    print(f"Is infinite: {np.isinf(result)}")
    print(f"Is NaN: {np.isnan(result)}")

    if w:
        print(f"\nWarnings raised ({len(w)}):")
        for warning in w:
            print(f"  - {warning.category.__name__}: {warning.message}")

# Test if ValueError is raised
print("\n5. Testing if ValueError is raised for invalid bases:")
print("-" * 40)

for test_base in [1.0, 0, -1]:
    try:
        result = jensenshannon(p, q, base=test_base)
        print(f"base={test_base}: No exception raised, result={result}")
    except ValueError as e:
        print(f"base={test_base}: ValueError raised: {e}")
    except Exception as e:
        print(f"base={test_base}: Other exception raised: {type(e).__name__}: {e}")

print("\n" + "=" * 60)