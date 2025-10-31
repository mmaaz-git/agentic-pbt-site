import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings
from xarray.core.variable import Variable
from xarray.coding.times import CFDatetimeCoder


@given(
    st.lists(
        st.datetimes(
            min_value=pd.Timestamp("2000-01-01"),
            max_value=pd.Timestamp("2050-12-31"),
        ),
        min_size=1,
        max_size=20,
    )
)
@settings(max_examples=100)
def test_datetime_coder_round_trip(datetime_list):
    datetime_arr = np.array(datetime_list, dtype="datetime64[ns]")

    encoding = {"units": "days since 2000-01-01", "calendar": "proleptic_gregorian"}
    original_var = Variable(("time",), datetime_arr, encoding=encoding)

    coder = CFDatetimeCoder()

    encoded_var = coder.encode(original_var)
    decoded_var = coder.decode(encoded_var)

    np.testing.assert_array_equal(original_var.data, decoded_var.data)
    assert original_var.dims == decoded_var.dims

if __name__ == "__main__":
    # Run the test - it should find a failing case
    test_datetime_coder_round_trip()
    print("Test completed without failures - this indicates the test may not be finding the bug!")