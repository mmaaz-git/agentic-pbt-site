#!/usr/bin/env python3
"""
Property-based test demonstrating the Django TemplateCommand --exclude bug.
"""
from hypothesis import given, strategies as st


@given(st.sets(st.text(min_size=1, max_size=20), min_size=1, max_size=5))
def test_exclude_preserves_dot_directory_filtering(exclude_dirs):
    """
    Test that using --exclude doesn't make filtering LESS restrictive.

    The bug: Without --exclude, all dot directories are excluded.
    With --exclude, only .git is excluded, allowing other dot dirs through.
    """
    # Test directories that might be in a project
    dirs_without_exclude = ['.git', '.vscode', '.idea', '__pycache__', 'mydir']
    dirs_with_exclude = dirs_without_exclude[:]

    # Build excluded_directories list as Django does
    excluded_directories = [".git", "__pycache__"] + list(exclude_dirs)

    # Simulate filtering WITHOUT --exclude (options dict has no "exclude" key)
    filtered_without = []
    for dirname in dirs_without_exclude:
        if not (dirname.startswith(".") or dirname == "__pycache__"):
            filtered_without.append(dirname)

    # Simulate filtering WITH --exclude (options dict has "exclude" key)
    filtered_with = []
    for dirname in dirs_with_exclude:
        if dirname not in excluded_directories:
            filtered_with.append(dirname)

    # Count dot directories that were NOT filtered out
    dot_dirs_without = [d for d in dirs_without_exclude if d.startswith('.') and d not in filtered_without]
    dot_dirs_with = [d for d in dirs_with_exclude if d.startswith('.') and d not in filtered_with]

    # The bug: Using --exclude should not make filtering LESS restrictive
    assert len(dot_dirs_with) >= len(dot_dirs_without), \
        f"Using --exclude should not make filtering LESS restrictive for dot directories. " \
        f"Without --exclude: {dot_dirs_without} excluded. " \
        f"With --exclude {exclude_dirs}: {dot_dirs_with} excluded."


if __name__ == "__main__":
    # Run the test - it should fail for most inputs
    test_exclude_preserves_dot_directory_filtering()