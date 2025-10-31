from hypothesis import given, strategies as st, settings
import pandas as pd
import numpy as np
from io import StringIO

@given(
    st.lists(
        st.tuples(st.integers(), st.integers()),
        min_size=1,
        max_size=50
    )
)
@settings(max_examples=100)
def test_dtype_specification_preserved(rows):
    csv_string = "a,b\n" + "\n".join(f"{a},{b}" for a, b in rows)
    result = pd.read_csv(StringIO(csv_string), dtype={'a': 'int64', 'b': 'int32'})

    assert result['a'].dtype == np.int64
    assert result['b'].dtype == np.int32

    # Check if any values in column b would overflow int32
    for i, (a, b) in enumerate(rows):
        if b > 2147483647 or b < -2147483648:
            print(f"Row {i}: Value {b} is out of int32 range")
            print(f"Parsed as: {result['b'].iloc[i]}")

print("Running Hypothesis test...")
test_dtype_specification_preserved()
print("Test completed without assertion errors")

print("\nTesting specific failing case from bug report:")
rows = [(0, 2147483648)]
csv_string = "a,b\n" + "\n".join(f"{a},{b}" for a, b in rows)
result = pd.read_csv(StringIO(csv_string), dtype={'a': 'int64', 'b': 'int32'})

print(f"Input: rows={(0, 2147483648)}")
print(f"Column 'a' dtype: {result['a'].dtype}")
print(f"Column 'b' dtype: {result['b'].dtype}")
print(f"Column 'b' value: {result['b'].iloc[0]} (expected 2147483648)")
print(f"Overflow detected: {result['b'].iloc[0] == -2147483648}")