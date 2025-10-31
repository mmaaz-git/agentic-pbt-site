from hypothesis import given, strategies as st, settings
import numpy as np
from xarray.core.variable import Variable
from xarray.coding.times import CFDatetimeCoder

@given(
    st.lists(
        st.integers(
            min_value=np.datetime64('1700-01-01', 'ns').astype('int64'),
            max_value=np.datetime64('2200-01-01', 'ns').astype('int64')
        ),
        min_size=1,
        max_size=50
    )
)
def test_datetime_coder_round_trip_ns(values):
    data = np.array(values, dtype='datetime64[ns]')
    original_var = Variable(('time',), data)

    coder = CFDatetimeCoder(use_cftime=False, time_unit='ns')

    encoded_var = coder.encode(original_var)
    decoded_var = coder.decode(encoded_var)

    decoded_data = decoded_var.data
    if hasattr(decoded_data, 'get_duck_array'):
        decoded_data = decoded_data.get_duck_array()

    np.testing.assert_array_equal(data, decoded_data)

# Run the test with the failing input from the report
if __name__ == "__main__":
    print("Testing with failing input: values=[703_036_036_854_775_809, -8_520_336_000_000_000_000]")
    values = [703_036_036_854_775_809, -8_520_336_000_000_000_000]
    try:
        # Run the test function directly with the values
        data = np.array(values, dtype='datetime64[ns]')
        original_var = Variable(('time',), data)
        coder = CFDatetimeCoder(use_cftime=False, time_unit='ns')
        encoded_var = coder.encode(original_var)
        decoded_var = coder.decode(encoded_var)
        decoded_data = decoded_var.data
        if hasattr(decoded_data, 'get_duck_array'):
            decoded_data = decoded_data.get_duck_array()
        np.testing.assert_array_equal(data, decoded_data)
        print("Test passed!")
    except Exception as e:
        import traceback
        print(f"Test failed with {type(e).__name__}: {e}")
        print("\nFull traceback:")
        traceback.print_exc()

    print("\nRunning property-based tests...")
    # Run the property test with limited examples
    test = test_datetime_coder_round_trip_ns

    # Run with settings decorator to limit number of examples
    test_limited = settings(max_examples=10)(test)
    try:
        test_limited()
        print("Property-based tests passed!")
    except Exception as e:
        print(f"Property-based tests failed: {e}")