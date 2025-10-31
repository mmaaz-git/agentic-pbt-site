"""Property-based test for Cython.Distutils.read_setup_file macro parsing bug."""

import tempfile
import os
from hypothesis import given, strategies as st, example
from Cython.Distutils.extension import read_setup_file


@given(
    st.text(min_size=1, max_size=20, alphabet=st.characters(min_codepoint=65, max_codepoint=90)),
    st.text(min_size=1, max_size=20, alphabet=st.characters(min_codepoint=48, max_codepoint=122))
    .filter(lambda x: '"' not in x and "'" not in x and ' ' not in x)
)
@example(macro_name='A', macro_value='0')  # A simple failing case
@example(macro_name='FOO', macro_value='bar')  # The example from documentation
def test_define_macro_value_parsing(macro_name, macro_value):
    """Test that -DNAME=value correctly defines NAME to value."""
    macro_def = f"{macro_name}={macro_value}"
    setup_line = f"testmod test.c -D{macro_def}"

    with tempfile.NamedTemporaryFile(mode='w', suffix='.setup', delete=False) as f:
        f.write(setup_line)
        f.flush()
        temp_path = f.name

    try:
        extensions = read_setup_file(temp_path)
        ext = extensions[0]

        assert len(ext.define_macros) == 1, f"Expected 1 macro, got {len(ext.define_macros)}"
        actual_name, actual_value = ext.define_macros[0]

        print(f"Input: -D{macro_name}={macro_value}")
        print(f"Expected: ('{macro_name}', '{macro_value}')")
        print(f"Actual:   ('{actual_name}', '{actual_value}')")

        assert actual_name == macro_name, f"Macro name mismatch: expected '{macro_name}', got '{actual_name}'"
        assert actual_value == macro_value, f"Macro value mismatch: expected '{macro_value}', got '{actual_value}'"
    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    # Run the test
    test_define_macro_value_parsing()