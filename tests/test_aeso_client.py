"""Unit tests for the AESO client. Uses mock HTTP responses."""
from __future__ import annotations

from datetime import date, datetime
from unittest.mock import Mock, patch

import pytest

from src.data.aeso_client import AESOClient, PoolPricePoint, _maybe_float


SAMPLE_AESO_RESPONSE = {
    "return": {
        "Pool Price Report": [
            {
                "begin_datetime_utc": "2024-06-12 06:00",
                "begin_datetime_mpt": "2024-06-12 00:00",
                "pool_price": "45.32",
                "forecast_pool_price": "43.21",
                "rolling_30day_avg": "38.15",
            },
            {
                "begin_datetime_utc": "2024-06-12 07:00",
                "begin_datetime_mpt": "2024-06-12 01:00",
                "pool_price": "42.10",
                "forecast_pool_price": "41.00",
                "rolling_30day_avg": "38.15",
            },
        ]
    }
}


class TestAESOClient:

    @patch("src.data.aeso_client.AESO_API_KEY", "")
    def test_requires_api_key(self):
        # Force the config-level AESO_API_KEY to "" so we can assert the
        # client refuses to start with no key at all — regardless of what
        # is in the developer's local .env file.
        with pytest.raises(ValueError, match="AESO_API_KEY is missing"):
            AESOClient(api_key="")

    def test_instantiates_with_key(self):
        client = AESOClient(api_key="dummy_key")
        assert client.api_key == "dummy_key"
        assert "API-KEY" in client._session.headers

    @patch("src.data.aeso_client.requests.Session.get")
    def test_fetch_and_parse(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = SAMPLE_AESO_RESPONSE
        mock_get.return_value = mock_response

        client = AESOClient(api_key="dummy_key")
        points = client.get_pool_prices(
            start_date=date(2024, 6, 12),
            use_cache=False,
        )

        assert len(points) == 2
        assert all(isinstance(p, PoolPricePoint) for p in points)
        assert points[0].pool_price == 45.32
        assert points[0].forecast_pool_price == 43.21
        assert points[0].rolling_30day_avg == 38.15
        assert points[0].begin_datetime_utc < points[1].begin_datetime_utc

    @patch("src.data.aeso_client.requests.Session.get")
    def test_retries_on_transient_error(self, mock_get):
        transient = Mock(status_code=503)
        ok = Mock(status_code=200)
        ok.json.return_value = SAMPLE_AESO_RESPONSE
        mock_get.side_effect = [transient, ok]

        client = AESOClient(api_key="dummy", max_retries=2, backoff_base_s=1.0)
        points = client.get_pool_prices(date(2024, 6, 12), use_cache=False)
        assert len(points) == 2
        assert mock_get.call_count == 2

    def test_to_dataframe(self):
        client = AESOClient(api_key="dummy")
        points = [
            PoolPricePoint(
                begin_datetime_utc=datetime(2024, 6, 12, 6, 0),
                begin_datetime_local=datetime(2024, 6, 12, 0, 0),
                pool_price=45.32,
                forecast_pool_price=43.21,
                rolling_30day_avg=38.15,
            ),
        ]
        df = client.to_dataframe(points)
        assert len(df) == 1
        assert "pool_price" in df.columns
        assert df["pool_price"].iloc[0] == 45.32

    def test_to_dataframe_empty(self):
        client = AESOClient(api_key="dummy")
        df = client.to_dataframe([])
        assert df.empty
        assert "pool_price" in df.columns

    def test_end_before_start_raises(self):
        client = AESOClient(api_key="dummy")
        with pytest.raises(ValueError, match="end_date must be >= start_date"):
            client.get_pool_prices(
                start_date=date(2024, 6, 12),
                end_date=date(2024, 6, 10),
            )


class TestMaybeFloat:

    @pytest.mark.parametrize("value,expected", [
        (None, None),
        ("", None),
        ("42.5", 42.5),
        (42.5, 42.5),
        ("not_a_number", None),
    ])
    def test_maybe_float(self, value, expected):
        assert _maybe_float(value) == expected
