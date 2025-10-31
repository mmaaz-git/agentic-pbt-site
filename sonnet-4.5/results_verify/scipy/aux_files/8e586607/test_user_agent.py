from unittest.mock import patch, MagicMock
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env/lib/python3.13/site-packages')
import scipy.datasets._fetchers
import scipy.datasets._download_all


def test_user_agent_consistency():
    """Both fetch_data and download_all should use the same User-Agent format."""
    with patch('scipy.datasets._fetchers.pooch') as mock_pooch_fetchers, \
         patch('scipy.datasets._download_all.pooch') as mock_pooch_download:

        mock_fetcher = MagicMock()
        mock_pooch_fetchers.HTTPDownloader.return_value = MagicMock()
        mock_pooch_download.HTTPDownloader.return_value = MagicMock()
        mock_pooch_download.os_cache.return_value = '/tmp/cache'
        mock_pooch_download.retrieve = MagicMock()

        scipy.datasets._fetchers.fetch_data("test.dat", data_fetcher=mock_fetcher)
        fetch_user_agent = mock_pooch_fetchers.HTTPDownloader.call_args[1]['headers']['User-Agent']

        scipy.datasets._download_all.download_all()
        download_user_agent = mock_pooch_download.HTTPDownloader.call_args[1]['headers']['User-Agent']

        print(f"fetch_data User-Agent: {fetch_user_agent}")
        print(f"download_all User-Agent: {download_user_agent}")

        assert fetch_user_agent == download_user_agent, f"User-Agent headers differ: fetch_data='{fetch_user_agent}' vs download_all='{download_user_agent}'"

if __name__ == "__main__":
    test_user_agent_consistency()