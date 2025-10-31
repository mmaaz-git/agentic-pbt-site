from hypothesis import given, strategies as st, settings
import numpy as np
import xarray as xr
from xarray.plot.utils import _rescale_imshow_rgb

@given(
    st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
        min_size=10,
        max_size=100
    )
)
@settings(max_examples=100)
def test_rescale_imshow_rgb_output_range_robust(data_list):
    data_values = np.array(data_list).reshape(-1, 1, 1)
    darray = xr.DataArray(data_values)

    result = _rescale_imshow_rgb(darray, None, None, robust=True)

    assert np.all(result >= 0), f"Found values < 0: min={result.min()}"
    assert np.all(result <= 1), f"Found values > 1: max={result.max()}"

# Test with the specific failing input
def test_specific_failing_case():
    data_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    data_values = np.array(data_list).reshape(-1, 1, 1)
    darray = xr.DataArray(data_values)

    result = _rescale_imshow_rgb(darray, None, None, robust=True)

    print(f"Result values: {result.values.flatten()}")
    print(f"Contains NaN: {np.any(np.isnan(result.values))}")
    print(f"Min value: {np.nanmin(result.values) if not np.all(np.isnan(result.values)) else 'All NaN'}")
    print(f"Max value: {np.nanmax(result.values) if not np.all(np.isnan(result.values)) else 'All NaN'}")

if __name__ == "__main__":
    print("Testing specific failing case:")
    test_specific_failing_case()

    print("\nRunning hypothesis tests:")
    test_rescale_imshow_rgb_output_range_robust()