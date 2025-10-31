from Cython.Build.Cache import Cache
import tempfile
import zipfile
import os

with tempfile.TemporaryDirectory() as tmpdir:
    cache = Cache(tmpdir)

    # Create a cached zip file with an artifact
    zip_path = os.path.join(tmpdir, 'cached.zip')
    with zipfile.ZipFile(zip_path, 'w') as z:
        z.writestr('output.c', 'int main() { return 0; }')

    # Create output directory structure
    output_dir = os.path.join(tmpdir, 'build')
    os.makedirs(output_dir)
    c_file = os.path.join(output_dir, 'test.c')

    # Load from cache
    cache.load_from_cache(c_file, zip_path)

    # Check where files actually ended up
    expected = os.path.join(output_dir, 'output.c')
    actual = os.path.join(output_dir, 'output.c', 'output.c')

    print(f'Expected location: {expected}')
    print(f'Exists as file: {os.path.isfile(expected)}')
    print(f'Exists as directory: {os.path.isdir(expected)}')
    print(f'\nActual location: {actual}')
    print(f'Exists as file: {os.path.isfile(actual)}')

    # Show directory structure
    print(f'\nDirectory structure in {output_dir}:')
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(output_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        sub_indent = ' ' * 2 * (level + 1)
        for file in files:
            print(f'{sub_indent}{file}')