"""
Tests for language code validation.
"""
import pytest

from app.services.language_codes import (
    is_valid_language,
    get_language_name,
    get_supported_languages,
)


class TestLanguageValidation:
    """Tests for language validation functions."""

    def test_valid_language_codes(self):
        """Test valid language codes."""
        assert is_valid_language("eng_Latn") is True
        assert is_valid_language("hin_Deva") is True
        assert is_valid_language("spa_Latn") is True
        assert is_valid_language("fra_Latn") is True
        assert is_valid_language("zho_Hans") is True

    def test_invalid_language_codes(self):
        """Test invalid language codes."""
        assert is_valid_language("invalid") is False
        assert is_valid_language("xyz_Abcd") is False
        assert is_valid_language("") is False

    def test_get_language_name(self):
        """Test getting language names."""
        assert get_language_name("eng_Latn") == "English"
        assert get_language_name("hin_Deva") == "Hindi"
        assert get_language_name("spa_Latn") == "Spanish"
        assert get_language_name("invalid") == "Unknown"

    def test_get_supported_languages(self):
        """Test getting supported languages."""
        languages = get_supported_languages()

        assert isinstance(languages, dict)
        assert len(languages) > 0
        assert "eng_Latn" in languages
        assert "hin_Deva" in languages

        # Check that all values are strings
        for code, name in languages.items():
            assert isinstance(code, str)
            assert isinstance(name, str)
            assert len(code) > 0
            assert len(name) > 0
