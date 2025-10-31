#!/usr/bin/env python3
"""Test to reproduce the dask.bag frequencies() bug"""

if __name__ == '__main__':
    import dask
    import dask.bag as db

    # Use single-threaded scheduler to avoid multiprocessing issues
    dask.config.set(scheduler='synchronous')

    # Test the actual functionality
    b = db.from_sequence(['Alice', 'Bob', 'Alice'])
    result = dict(b.frequencies())
    print(f"Actual result from frequencies(): {result}")

    # Show the docstring syntax error
    docstring_example = "{'Alice': 2, 'Bob', 1}"
    print(f"\nDocstring example: {docstring_example}")

    # Try to evaluate the docstring example
    try:
        eval(docstring_example)
        print("Docstring example is valid Python syntax")
    except SyntaxError as e:
        print(f"Docstring example has syntax error: {e}")

    # Show what it should be
    correct_example = "{'Alice': 2, 'Bob': 1}"
    print(f"\nCorrect syntax: {correct_example}")
    try:
        correct = eval(correct_example)
        print(f"Evaluates to: {correct}")
    except SyntaxError as e:
        print(f"Error: {e}")