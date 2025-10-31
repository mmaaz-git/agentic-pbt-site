import sys
import scipy
import scipy.datasets._fetchers as fetchers
import scipy.datasets._download_all as download_all_module
import inspect


def test_user_agent_inconsistency_demonstration():
    fetch_data_source = inspect.getsource(fetchers.fetch_data)
    download_all_source = inspect.getsource(download_all_module.download_all)

    assert 'f"SciPy {sys.modules[\'scipy\'].__version__}"' in fetch_data_source or \
           "f\"SciPy {sys.modules['scipy'].__version__}\"" in fetch_data_source

    assert '"SciPy"' in download_all_source and \
           '__version__' not in download_all_source

    print("Test passed - inconsistency confirmed")

test_user_agent_inconsistency_demonstration()