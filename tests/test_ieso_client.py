"""Unit tests for the IESO client. Uses sample CSV strings."""
from __future__ import annotations

from datetime import date
from unittest.mock import Mock, patch

import pytest

from src.data.ieso_client import (
    IESOClient,
    OntarioPricePoint,
    HOEP_RETIREMENT_DATE,
)


# Realistic sample of the IESO Real-Time Market Price CSV (post-May-2025 format).
SAMPLE_OEMP_CSV = """CREATED AT,2025-06-15
DELIVERY_DATE,DELIVERY_HOUR,INTERVAL,PRICE
2025-06-15,1,1,34.20
2025-06-15,1,2,35.10
2025-06-15,1,3,33.80
2025-06-15,2,1,28.50
2025-06-15,2,2,29.00
2025-06-15,3,1,150.00
"""


# Sample of the legacy HOEP daily CSV
SAMPLE_HOEP_CSV = """Hour,HOEP
1,25.30
2,22.10
3,20.50
"""


class TestIESOClient:

    def test_instantiates_without_key(self):
        client = IESOClient()
        assert client._session is not None

    def test_range_end_before_start_raises(self):
        client = IESOClient()
        with pytest.raises(ValueError, match="end_date must be >= start_date"):
            client.get_range(date(2025, 6, 15), date(2025, 6, 10))

    @patch("src.data.ieso_client.requests.Session.get")
    def test_fetch_oemp_post_retirement(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = SAMPLE_OEMP_CSV
        mock_get.return_value = mock_response

        client = IESOClient()
        points = client.get_ontario_prices(date(2025, 6, 15), use_cache=False)

        assert len(points) == 3
        assert all(isinstance(p, OntarioPricePoint) for p in points)
        assert all(p.price_type == "OEMP" for p in points)

        # Hour 1: (34.20 + 35.10 + 33.80) / 3 = 34.3666...
        h1 = next(p for p in points if p.hour == 1)
        assert abs(h1.price - 34.3666) < 0.01

        # Hour 3: single value 150.00 (a spike)
        h3 = next(p for p in points if p.hour == 3)
        assert h3.price == 150.00

    @patch("src.data.ieso_client.requests.Session.get")
    def test_fetch_hoep_pre_retirement(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = SAMPLE_HOEP_CSV
        mock_get.return_value = mock_response

        client = IESOClient()
        points = client.get_ontario_prices(date(2024, 12, 1), use_cache=False)

        assert len(points) == 3
        assert all(p.price_type == "HOEP" for p in points)

    @patch("src.data.ieso_client.requests.Session.get")
    def test_returns_empty_on_404(self, mock_get):
        # IESO returns 404 for dates not yet published
        mock_response = Mock(status_code=404)
        mock_get.return_value = mock_response

        client = IESOClient()
        points = client.get_ontario_prices(date(2099, 1, 1), use_cache=False)
        assert points == []

    def test_to_dataframe_empty(self):
        client = IESOClient()
        df = client.to_dataframe([])
        assert df.empty


class TestHOEPRetirement:
    """Verifies we correctly route pre-/post-retirement dates."""

    def test_retirement_boundary(self):
        assert HOEP_RETIREMENT_DATE == date(2025, 5, 1)

    def test_post_retirement_uses_oemp(self):
        assert date(2025, 5, 15) >= HOEP_RETIREMENT_DATE

    def test_pre_retirement_uses_hoep(self):
        assert date(2025, 4, 30) < HOEP_RETIREMENT_DATE
