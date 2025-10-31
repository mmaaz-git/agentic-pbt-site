import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st

@given(
    st.text(min_size=1, max_size=5),
    st.text(min_size=1, max_size=2),
    st.text(min_size=2, max_size=10)
)
def test_replace_preserves_content(s, old, new):
    arr = np.array([s])
    numpy_result = nps.replace(arr, old, new)[0]
    python_result = s.replace(old, new)
    assert str(numpy_result) == python_result

# Run the specific failing example
print("Testing specific failing example from bug report:")
s = '0'
arr = np.array([s])
numpy_result = nps.replace(arr, '0', '00')[0]
python_result = s.replace('0', '00')
print(f"Input: '{s}'")
print(f"Replace '0' with '00'")
print(f"NumPy result: '{numpy_result}'")
print(f"Python result: '{python_result}'")
print(f"Match: {str(numpy_result) == python_result}")

# Run hypothesis test
if __name__ == "__main__":
    test_replace_preserves_content()