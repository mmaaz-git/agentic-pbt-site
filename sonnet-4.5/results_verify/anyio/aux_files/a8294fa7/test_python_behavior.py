import tempfile


def test_python_file_read():
    """Test Python's file.read() behavior with different values"""
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        test_data = b'x' * 1000
        f.write(test_data)
        f.flush()
        path = f.name

    try:
        # Test with -1
        with open(path, 'rb') as f:
            data = f.read(-1)
            print(f"file.read(-1): {len(data)} bytes - SUCCESS")

        # Test with 0
        with open(path, 'rb') as f:
            data = f.read(0)
            print(f"file.read(0): {len(data)} bytes - SUCCESS")

        # Test with positive value
        with open(path, 'rb') as f:
            data = f.read(100)
            print(f"file.read(100): {len(data)} bytes - SUCCESS")

        # Test with other negative values
        test_values = [-2, -10, -100]
        for val in test_values:
            try:
                with open(path, 'rb') as f:
                    data = f.read(val)
                    print(f"file.read({val}): {len(data)} bytes - SUCCESS")
            except ValueError as e:
                print(f"file.read({val}): ValueError - {e}")

    finally:
        import os
        os.unlink(path)


if __name__ == "__main__":
    test_python_file_read()