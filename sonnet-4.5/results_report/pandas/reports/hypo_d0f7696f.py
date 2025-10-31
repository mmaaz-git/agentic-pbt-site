from hypothesis import given, settings, strategies as st
import string
import pytest
from pandas.io.sas import read_sas

@given(
    base_name=st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=20),
    middle_ext=st.sampled_from(['.xpt', '.sas7bdat']),
    suffix=st.text(alphabet=string.ascii_lowercase + string.digits, min_size=1, max_size=10)
)
@settings(max_examples=200)
def test_format_detection_false_positives(base_name, middle_ext, suffix):
    filename = base_name + middle_ext + "." + suffix

    with pytest.raises((ValueError, FileNotFoundError, OSError)):
        read_sas(filename)

if __name__ == "__main__":
    test_format_detection_false_positives()