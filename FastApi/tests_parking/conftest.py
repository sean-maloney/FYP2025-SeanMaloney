import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock

from app_parking.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c
