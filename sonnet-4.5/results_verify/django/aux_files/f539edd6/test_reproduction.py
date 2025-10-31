#!/usr/bin/env python3
"""Test script to reproduce the FileSystemStorage line ending bug"""

import os
import sys
import tempfile
from pathlib import Path

# Add Django env to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

# Setup Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django.conf.global_settings')
import django
from django.conf import settings
if not settings.configured:
    settings.configure(
        DEBUG=True,
        INSTALLED_APPS=['django.contrib.contenttypes'],
    )
django.setup()

from django.core.files.base import ContentFile
from django.core.files.storage import FileSystemStorage
from django.core.files.storage.memory import InMemoryStorage


def test_hypothesis_case():
    """Run the property-based test from the bug report"""
    print("=" * 60)
    print("Testing Hypothesis Property-Based Test Case")
    print("=" * 60)

    from hypothesis import given, settings as hypothesis_settings, strategies as st

    @st.composite
    def file_contents(draw):
        content_type = draw(st.sampled_from(['bytes', 'text']))
        if content_type == 'bytes':
            return draw(st.binary(min_size=0, max_size=10000))
        else:
            return draw(st.text(min_size=0, max_size=10000))

    @given(file_contents())
    @hypothesis_settings(max_examples=200)
    def test_save_open_roundtrip_filesystem(content):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileSystemStorage(location=tmpdir)
            file_obj = ContentFile(content, name='test.txt')
            saved_name = storage.save('test.txt', file_obj)

            with storage.open(saved_name, 'rb' if isinstance(content, bytes) else 'r') as f:
                retrieved_content = f.read()

            assert retrieved_content == content, f"Content mismatch: {repr(content)} != {repr(retrieved_content)}"

    try:
        test_save_open_roundtrip_filesystem()
        print("Test passed!")
    except Exception as e:
        print(f"Test failed with error: {e}")


def test_specific_cases():
    """Test specific failing cases mentioned in the bug report"""
    print("\n" + "=" * 60)
    print("Testing Specific Line Ending Cases")
    print("=" * 60)

    test_cases = [
        '\r',
        '\r\n',
        'hello\rworld',
        'hello\r\nworld',
        'line1\rline2\rline3',
        'windows\r\nstyle\r\nfile'
    ]

    for content in test_cases:
        print(f"\nTesting content: {repr(content)}")

        with tempfile.TemporaryDirectory() as tmpdir:
            fs_storage = FileSystemStorage(location=tmpdir)
            mem_storage = InMemoryStorage()

            fs_file = ContentFile(content, name='test.txt')
            mem_file = ContentFile(content, name='test.txt')

            fs_saved = fs_storage.save('test.txt', fs_file)
            mem_saved = mem_storage.save('test.txt', mem_file)

            # Test text mode
            with fs_storage.open(fs_saved, 'r') as f:
                fs_result = f.read()

            with mem_storage.open(mem_saved, 'r') as f:
                mem_result = f.read()

            print(f"  Original:      {repr(content)}")
            print(f"  FS result:     {repr(fs_result)}")
            print(f"  Mem result:    {repr(mem_result)}")
            print(f"  FS == Mem:     {fs_result == mem_result}")
            print(f"  FS == Orig:    {fs_result == content}")
            print(f"  Mem == Orig:   {mem_result == content}")

            # Also test binary mode
            with fs_storage.open(fs_saved, 'rb') as f:
                fs_binary = f.read()

            with mem_storage.open(mem_saved, 'rb') as f:
                mem_binary = f.read()

            print(f"  FS (binary):   {repr(fs_binary)}")
            print(f"  Mem (binary):  {repr(mem_binary)}")
            print(f"  Binary match:  {fs_binary == mem_binary}")


def main():
    print("Testing Django FileSystemStorage Line Ending Bug")
    print("=" * 60)

    # Test hypothesis case
    try:
        test_hypothesis_case()
    except ImportError:
        print("Hypothesis not installed, skipping property-based test")

    # Test specific cases
    test_specific_cases()


if __name__ == "__main__":
    main()