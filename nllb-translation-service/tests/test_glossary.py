"""
Tests for glossary processing.
"""
import pytest

from app.glossary.processor import GlossaryProcessor


class TestGlossaryProcessor:
    """Tests for glossary processor."""

    @pytest.fixture
    def processor(self):
        """Create glossary processor."""
        return GlossaryProcessor()

    def test_apply_glossary_simple(self, processor):
        """Test simple glossary application."""
        translated = "The patient has high blood pressure"
        original = "The patient has hypertension"
        glossary = {"hypertension": "high blood pressure"}

        result, applied = processor.apply_glossary(
            translated_text=translated,
            original_text=original,
            glossary=glossary,
        )

        assert applied is False  # Term not in translated text
        assert isinstance(result, str)

    def test_apply_glossary_with_match(self, processor):
        """Test glossary with matching terms."""
        translated = "The patient has hypertension"
        original = "The patient has hypertension"
        glossary = {"hypertension": "उच्च रक्तचाप"}

        result, applied = processor.apply_glossary(
            translated_text=translated,
            original_text=original,
            glossary=glossary,
        )

        assert applied is True
        assert "उच्च रक्तचाप" in result

    def test_apply_empty_glossary(self, processor):
        """Test with empty glossary."""
        translated = "Some text"
        original = "Some text"
        glossary = {}

        result, applied = processor.apply_glossary(
            translated_text=translated,
            original_text=original,
            glossary=glossary,
        )

        assert applied is False
        assert result == translated

    def test_apply_none_glossary(self, processor):
        """Test with None glossary."""
        translated = "Some text"
        original = "Some text"

        result, applied = processor.apply_glossary(
            translated_text=translated,
            original_text=original,
            glossary=None,
        )

        assert applied is False
        assert result == translated

    def test_validate_glossary(self, processor):
        """Test glossary validation."""
        # Valid glossary
        assert processor.validate_glossary({"key": "value"}) is True
        assert processor.validate_glossary({}) is True

        # Invalid glossary
        assert processor.validate_glossary({"key": ""}) is False
        assert processor.validate_glossary({"": "value"}) is False
        assert processor.validate_glossary("not a dict") is False
        assert processor.validate_glossary({"key": 123}) is False

    def test_merge_glossaries(self, processor):
        """Test merging glossaries."""
        glossary1 = {"term1": "translation1", "term2": "translation2"}
        glossary2 = {"term2": "new_translation2", "term3": "translation3"}

        merged = processor.merge_glossaries(glossary1, glossary2)

        assert len(merged) == 3
        assert merged["term1"] == "translation1"
        assert merged["term2"] == "new_translation2"  # Later takes precedence
        assert merged["term3"] == "translation3"

    def test_merge_with_none(self, processor):
        """Test merging with None glossaries."""
        glossary1 = {"term1": "translation1"}

        merged = processor.merge_glossaries(glossary1, None)

        assert merged == glossary1

        merged = processor.merge_glossaries(None, glossary1)

        assert merged == glossary1

    def test_case_sensitive_matching(self, processor):
        """Test case-sensitive glossary matching."""
        translated = "Hypertension is common"
        original = "Hypertension is common"
        glossary = {"hypertension": "उच्च रक्तचाप"}

        # Case-insensitive (default)
        result, applied = processor.apply_glossary(
            translated_text=translated,
            original_text=original,
            glossary=glossary,
            case_sensitive=False,
        )

        assert applied is True

        # Case-sensitive
        result, applied = processor.apply_glossary(
            translated_text=translated,
            original_text=original,
            glossary=glossary,
            case_sensitive=True,
        )

        assert applied is False  # "Hypertension" != "hypertension"
