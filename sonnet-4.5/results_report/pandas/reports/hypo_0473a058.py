from hypothesis import given, strategies as st, settings
import tempfile
import os
from pandas.io.sas import read_sas

@given(
    extension=st.sampled_from(['.xpt', '.sas7bdat']),
    suffix=st.text(st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')),
                   min_size=1, max_size=10)
)
@settings(max_examples=100)
def test_format_detection_embedded_extension(extension, suffix):
    filename = f"file{extension}{suffix}"

    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{filename}') as tmp:
        tmp_path = tmp.name

    try:
        read_sas(tmp_path)
        assert False, f"Should have failed for {tmp_path}"
    except ValueError as e:
        error_msg = str(e).lower()
        if "unable to infer format" in error_msg:
            pass
        elif "header record" in error_msg:
            assert False, f"BUG: File '{filename}' was incorrectly detected as SAS format!"
        elif "magic number" in error_msg:
            assert False, f"BUG: File '{filename}' was incorrectly detected as SAS format!"
    finally:
        os.unlink(tmp_path)

# Run the test
test_format_detection_embedded_extension()