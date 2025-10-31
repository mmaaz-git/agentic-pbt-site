from hypothesis import given, strategies as st, example
from pandas.io.excel._openpyxl import OpenpyxlWriter

@given(
    ext=st.text(alphabet='abcdefghijklmnopqrstuvwxyz', min_size=1, max_size=3).filter(
        lambda x: x not in ['xlsx', 'xlsm', 'xls', 'xlsb', 'ods', 'odt', 'odf']
    )
)
@example(ext='l')
@example(ext='x')
@example(ext='s')
@example(ext='m')
def test_check_extension_substring_bug(ext):
    try:
        result = OpenpyxlWriter.check_extension(f'.{ext}')
        supported = OpenpyxlWriter._supported_extensions
        is_substring_of_any = any(ext in extension for extension in supported)
        is_exact_match = any(f'.{ext}' == extension for extension in supported)

        if not is_exact_match and is_substring_of_any:
            print(f"Bug found: '.{ext}' was accepted but is not a supported extension.")
            print(f"  It's a substring of {[e for e in supported if ext in e]} but not an exact match.")
            raise AssertionError(
                f"Bug found: '.{ext}' was accepted but is not a supported extension. "
                f"It's a substring of {[e for e in supported if ext in e]} but not an exact match."
            )
    except ValueError:
        pass  # This is expected for truly unsupported extensions

# Run the test
test_check_extension_substring_bug()