"""
Tests for API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health endpoint."""

    @patch("app.api.routes.get_translation_service")
    def test_health_check(self, mock_get_service, client):
        """Test health check endpoint."""
        # Mock service
        mock_service = MagicMock()
        mock_service.model_loaded = True
        mock_service.is_gpu_available.return_value = False
        mock_get_service.return_value = mock_service

        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert "model_loaded" in data
        assert "gpu_available" in data


class TestLanguagesEndpoint:
    """Tests for languages endpoint."""

    def test_get_languages(self, client):
        """Test get languages endpoint."""
        response = client.get("/api/v1/languages")

        assert response.status_code == 200
        data = response.json()
        assert "languages" in data
        assert "total" in data
        assert isinstance(data["languages"], list)
        assert data["total"] > 0

        # Check language structure
        if len(data["languages"]) > 0:
            lang = data["languages"][0]
            assert "code" in lang
            assert "name" in lang


class TestTranslationEndpoint:
    """Tests for translation endpoint."""

    @patch("app.api.routes.get_translation_service")
    @patch("app.api.routes.get_glossary_processor")
    def test_translate_success(
        self, mock_get_glossary, mock_get_service, client
    ):
        """Test successful translation."""
        # Mock translation service
        mock_service = MagicMock()
        mock_service.translate.return_value = "नमस्ते"
        mock_get_service.return_value = mock_service

        # Mock glossary processor
        mock_glossary = MagicMock()
        mock_glossary.apply_glossary.return_value = ("नमस्ते", False)
        mock_get_glossary.return_value = mock_glossary

        # Make request
        response = client.post(
            "/api/v1/translate",
            json={
                "text": "Hello",
                "source_lang": "eng_Latn",
                "target_lang": "hin_Deva",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "translated_text" in data
        assert data["source_lang"] == "eng_Latn"
        assert data["target_lang"] == "hin_Deva"
        assert "glossary_applied" in data

    @patch("app.api.routes.get_translation_service")
    @patch("app.api.routes.get_glossary_processor")
    def test_translate_with_glossary(
        self, mock_get_glossary, mock_get_service, client
    ):
        """Test translation with glossary."""
        # Mock translation service
        mock_service = MagicMock()
        mock_service.translate.return_value = "Initial translation"
        mock_get_service.return_value = mock_service

        # Mock glossary processor
        mock_glossary = MagicMock()
        mock_glossary.apply_glossary.return_value = (
            "Translation with glossary",
            True,
        )
        mock_get_glossary.return_value = mock_glossary

        # Make request with glossary
        response = client.post(
            "/api/v1/translate",
            json={
                "text": "Hello",
                "source_lang": "eng_Latn",
                "target_lang": "hin_Deva",
                "glossary": {"Hello": "नमस्ते"},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["glossary_applied"] is True

    def test_translate_invalid_request(self, client):
        """Test translation with invalid request."""
        response = client.post(
            "/api/v1/translate",
            json={
                "text": "",  # Empty text
                "source_lang": "eng_Latn",
                "target_lang": "hin_Deva",
            },
        )

        assert response.status_code == 422  # Validation error

    @patch("app.api.routes.get_translation_service")
    def test_translate_invalid_language(self, mock_get_service, client):
        """Test translation with invalid language code."""
        # Mock service to raise ValueError
        mock_service = MagicMock()
        mock_service.translate.side_effect = ValueError("Invalid language")
        mock_get_service.return_value = mock_service

        response = client.post(
            "/api/v1/translate",
            json={
                "text": "Hello",
                "source_lang": "invalid_code",
                "target_lang": "hin_Deva",
            },
        )

        assert response.status_code == 400


class TestBatchTranslationEndpoint:
    """Tests for batch translation endpoint."""

    @patch("app.api.routes.get_translation_service")
    @patch("app.api.routes.get_glossary_processor")
    def test_batch_translate(
        self, mock_get_glossary, mock_get_service, client
    ):
        """Test batch translation."""
        # Mock translation service
        mock_service = MagicMock()
        mock_service.translate.side_effect = ["नमस्ते", "Hola"]
        mock_get_service.return_value = mock_service

        # Mock glossary processor
        mock_glossary = MagicMock()
        mock_glossary.apply_glossary.return_value = ("Translation", False)
        mock_get_glossary.return_value = mock_glossary

        # Make batch request
        response = client.post(
            "/api/v1/batch-translate",
            json={
                "items": [
                    {
                        "text": "Hello",
                        "source_lang": "eng_Latn",
                        "target_lang": "hin_Deva",
                    },
                    {
                        "text": "Hello",
                        "source_lang": "eng_Latn",
                        "target_lang": "spa_Latn",
                    },
                ]
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "translations" in data
        assert "total" in data
        assert len(data["translations"]) == 2
        assert data["total"] == 2

    def test_batch_translate_empty(self, client):
        """Test batch translation with empty items."""
        response = client.post(
            "/api/v1/batch-translate",
            json={"items": []},
        )

        assert response.status_code == 422  # Validation error
