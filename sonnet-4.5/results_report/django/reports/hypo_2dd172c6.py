from hypothesis import given, strategies as st, settings, assume
from django.apps.config import AppConfig


@given(st.text(alphabet='abcdefghijklmnopqrstuvwxyz.', min_size=1, max_size=30))
@settings(max_examples=1000)
def test_create_rpartition_edge_cases(entry):
    mod_path, _, cls_name = entry.rpartition(".")

    if mod_path and not cls_name:
        try:
            config = AppConfig.create(entry)
        except IndexError as e:
            print(f"Failed on input: {repr(entry)}")
            print(f"IndexError: {e}")
            assert False, f"IndexError should not occur: {e}"
        except Exception:
            pass

if __name__ == "__main__":
    test_create_rpartition_edge_cases()