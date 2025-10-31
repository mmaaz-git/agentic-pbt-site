from hypothesis import given, strategies as st
from Cython.Build.Dependencies import normalize_existing
import tempfile
import os

# First, let's reproduce the simple example
print("=== Reproducing the simple example ===")
with tempfile.TemporaryDirectory() as tmpdir:
    base_file = os.path.join(tmpdir, 'base.txt')
    with open(base_file, 'w') as f:
        f.write('base')

    existing_file = os.path.join(tmpdir, 'test.txt')
    with open(existing_file, 'w') as f:
        f.write('test')

    paths = ['test.txt', existing_file]

    normalized, base_dir = normalize_existing(base_file, paths)

    print(f'Input: {paths}')
    print(f'Output: {normalized}')
    print(f'Unique: {set(normalized)}')
    print(f'Length of output: {len(normalized)}')
    print(f'Length of unique: {len(set(normalized))}')
    print(f'Has duplicates: {len(normalized) != len(set(normalized))}')

# Now let's run the property-based test
print("\n=== Running the property-based test ===")

valid_filename = st.text(
    alphabet=st.characters(blacklist_categories=('Cs',), blacklist_characters='\x00/'),
    min_size=1,
    max_size=20
).filter(lambda s: '{' not in s and '}' not in s and ',' not in s and '\x00' not in s and s not in ['.', '..'])

@given(valid_filename)
def test_normalize_existing_no_duplicates(filename):
    with tempfile.TemporaryDirectory() as tmpdir:
        base_file = os.path.join(tmpdir, 'base.txt')
        with open(base_file, 'w') as f:
            f.write('base')

        existing_file = os.path.join(tmpdir, filename)
        with open(existing_file, 'w') as f:
            f.write('test')

        paths = [filename, existing_file]

        normalized, base_dir = normalize_existing(base_file, paths)

        assert len(normalized) == len(set(normalized)), f"Should not produce duplicates. Got {normalized} from {paths}"

# Run the test
try:
    test_normalize_existing_no_duplicates()
    print("Property-based test passed!")
except AssertionError as e:
    print(f"Property-based test failed: {e}")
except Exception as e:
    print(f"Property-based test error: {e}")