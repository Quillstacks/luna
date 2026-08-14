import pytest
import requests
from unittest.mock import MagicMock, patch
from pathlib import Path

from luna.io.pds_index import PDSIndex
from luna.io.pds_fetch import fetch_nac
from luna.exceptions import IndexError as LunaIndexError, NACNotFoundError


def test_pds_index_read_record_retry_on_429():
    """Test that _read_record retries on HTTP 429 and succeeds when response recovers."""
    index = PDSIndex()
    mock_resp_429 = MagicMock()
    mock_resp_429.status_code = 429

    mock_resp_206 = MagicMock()
    mock_resp_206.status_code = 206
    mock_resp_206.content = b"TEST_RECORD_CONTENT_HERE"

    index.session.get = MagicMock(side_effect=[mock_resp_429, mock_resp_206])

    with patch("luna.io.pds_index.time.sleep") as mock_sleep:
        record = index._read_record("https://example.com/INDEX.TAB", row=0, record_bytes=24)
        assert record == "TEST_RECORD_CONTENT_HERE"
        assert index.session.get.call_count == 2
        mock_sleep.assert_called_once()


def test_pds_index_read_record_exhausts_retries():
    """Test that _read_record raises LunaIndexError when 429 persists across all retries."""
    index = PDSIndex()
    mock_resp_429 = MagicMock()
    mock_resp_429.status_code = 429

    index.session.get = MagicMock(return_value=mock_resp_429)

    with patch("luna.io.pds_index.time.sleep"):
        with pytest.raises(LunaIndexError, match="HTTP 429 on range read"):
            index._read_record("https://example.com/INDEX.TAB", row=0, record_bytes=24)
        assert index.session.get.call_count == 5


def test_pds_fetch_retry_on_http_error(tmp_path: Path):
    """Test that fetch_nac retries on requests.exceptions.HTTPError (e.g. 429/503)."""
    mock_response_fail = MagicMock()
    http_err = requests.exceptions.HTTPError("429 Too Many Requests", response=mock_response_fail)

    mock_response_ok = MagicMock()
    mock_response_ok.__enter__.return_value = mock_response_ok
    mock_response_ok.headers = {"Content-Length": "12"}
    mock_response_ok.iter_content = MagicMock(return_value=[b"MOCK_IMG_DATA"])
    mock_response_ok.raise_for_status = MagicMock()

    mock_fail_cm = MagicMock()
    mock_fail_cm.__enter__.side_effect = http_err

    with patch("requests.get", side_effect=[mock_fail_cm, mock_response_ok]), \
         patch("time.sleep"):
        out_path = fetch_nac("M126710873RE", dest_dir=tmp_path, url="https://example.com/M126710873RE.IMG", retries=3)
        assert out_path.exists()
        assert out_path.read_bytes() == b"MOCK_IMG_DATA"
