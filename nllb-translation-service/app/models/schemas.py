"""
Pydantic schemas for API requests and responses.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, List


class TranslationRequest(BaseModel):
    """Single translation request schema."""

    text: str = Field(..., description="Text to translate", min_length=1, max_length=5000)
    source_lang: str = Field(..., description="Source language code (e.g., eng_Latn)")
    target_lang: str = Field(..., description="Target language code (e.g., hin_Deva)")
    glossary: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional glossary for term translation overrides"
    )

    @field_validator("source_lang", "target_lang")
    @classmethod
    def validate_language_code(cls, v: str) -> str:
        """Validate language code format."""
        if not v or len(v) < 3:
            raise ValueError("Language code must be at least 3 characters")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Hello, how are you?",
                "source_lang": "eng_Latn",
                "target_lang": "hin_Deva",
                "glossary": {
                    "hello": "नमस्ते"
                }
            }
        }


class TranslationResponse(BaseModel):
    """Single translation response schema."""

    translated_text: str = Field(..., description="Translated text")
    source_lang: str = Field(..., description="Source language code")
    target_lang: str = Field(..., description="Target language code")
    glossary_applied: bool = Field(
        default=False,
        description="Whether glossary terms were applied"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "translated_text": "नमस्ते, आप कैसे हैं?",
                "source_lang": "eng_Latn",
                "target_lang": "hin_Deva",
                "glossary_applied": True
            }
        }


class BatchTranslationItem(BaseModel):
    """Single item in batch translation request."""

    text: str = Field(..., min_length=1, max_length=5000)
    source_lang: str
    target_lang: str


class BatchTranslationRequest(BaseModel):
    """Batch translation request schema."""

    items: List[BatchTranslationItem] = Field(
        ...,
        description="List of translation items",
        min_length=1,
        max_length=100
    )
    glossary: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional glossary applied to all translations"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "items": [
                    {
                        "text": "Hello",
                        "source_lang": "eng_Latn",
                        "target_lang": "hin_Deva"
                    },
                    {
                        "text": "Good morning",
                        "source_lang": "eng_Latn",
                        "target_lang": "spa_Latn"
                    }
                ]
            }
        }


class BatchTranslationResponse(BaseModel):
    """Batch translation response schema."""

    translations: List[TranslationResponse] = Field(
        ...,
        description="List of translation results"
    )
    total: int = Field(..., description="Total number of translations")


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Service version")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    gpu_available: bool = Field(..., description="Whether GPU is available")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "model_loaded": True,
                "gpu_available": True
            }
        }


class LanguageInfo(BaseModel):
    """Language information schema."""

    code: str = Field(..., description="Language code")
    name: str = Field(..., description="Language name")


class LanguagesResponse(BaseModel):
    """Supported languages response schema."""

    languages: List[LanguageInfo] = Field(..., description="List of supported languages")
    total: int = Field(..., description="Total number of supported languages")


class ErrorResponse(BaseModel):
    """Error response schema."""

    detail: str = Field(..., description="Error message")
    error_type: Optional[str] = Field(default=None, description="Error type")
    request_id: Optional[str] = Field(default=None, description="Request ID for tracking")

    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Translation failed",
                "error_type": "TranslationError",
                "request_id": "abc-123-def"
            }
        }
