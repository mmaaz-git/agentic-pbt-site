from hypothesis import given, strategies as st, settings
import string
from pandas.io.sas import read_sas


@given(
    base=st.text(
        alphabet=string.ascii_letters + string.digits,
        min_size=1, max_size=20
    ),
    extension=st.sampled_from([".txt", ".csv", ".dat", ".backup", ".old", ".tmp"])
)
@settings(max_examples=500)
def test_format_detection_should_check_extension_not_substring(base, extension):
    filename = f"{base}.xpt{extension}"

    try:
        read_sas(filename)
        format_detected = True
    except FileNotFoundError:
        format_detected = True
    except ValueError as e:
        if "unable to infer format" in str(e):
            format_detected = False
        else:
            raise

    assert not format_detected, \
        f"File '{filename}' should not be detected as xport format (extension is {extension}, not .xpt)"

if __name__ == "__main__":
    test_format_detection_should_check_extension_not_substring()