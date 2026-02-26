"""
Pytest configuration and fixtures.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, MagicMock

from app.main import app
from app.services import TranslationService


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_translation_service():
    """Create mock translation service."""
    service = Mock(spec=TranslationService)
    service.model_loaded = True
    service.device = "cpu"
    service.translate = MagicMock(return_value="Translated text")
    service.is_gpu_available = MagicMock(return_value=False)
    service.validate_languages = MagicMock(return_value=None)
    return service
