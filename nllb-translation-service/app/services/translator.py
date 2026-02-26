"""
NLLB Translation Service.
Handles model loading, caching, and translation operations.
"""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from typing import Optional
import os
from functools import lru_cache

from ..core import get_settings, get_logger
from .language_codes import is_valid_language, get_language_name

logger = get_logger(__name__)
settings = get_settings()


class TranslationService:
    """Service for handling NLLB model and translation operations."""

    def __init__(self):
        """Initialize the translation service."""
        self.model: Optional[AutoModelForSeq2SeqLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.device: str = "cpu"
        self.model_loaded: bool = False

    def load_model(self) -> None:
        """Load NLLB model and tokenizer."""
        try:
            logger.info(f"Loading model: {settings.MODEL_NAME}")

            # Determine model path
            model_path = (
                settings.CUSTOM_MODEL_PATH
                if settings.CUSTOM_MODEL_PATH and os.path.exists(settings.CUSTOM_MODEL_PATH)
                else settings.MODEL_NAME
            )

            # Set device
            if settings.USE_GPU and torch.cuda.is_available():
                self.device = "cuda"
                logger.info("Using GPU for inference")
            else:
                self.device = "cpu"
                logger.info("Using CPU for inference")

            # Create cache directory if it doesn't exist
            os.makedirs(settings.MODEL_CACHE_DIR, exist_ok=True)

            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                cache_dir=settings.MODEL_CACHE_DIR,
                src_lang="eng_Latn",  # Default, will be overridden
            )

            # Load model
            logger.info("Loading model...")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                cache_dir=settings.MODEL_CACHE_DIR,
            )

            # Move model to device
            self.model.to(self.device)

            # Enable half precision if GPU and enabled
            if self.device == "cuda" and settings.USE_HALF_PRECISION:
                logger.info("Enabling half precision (FP16)")
                self.model.half()

            # Set model to evaluation mode
            self.model.eval()

            self.model_loaded = True
            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")

    def validate_languages(self, source_lang: str, target_lang: str) -> None:
        """
        Validate source and target language codes.

        Args:
            source_lang: Source language code
            target_lang: Target language code

        Raises:
            ValueError: If language codes are invalid
        """
        if not is_valid_language(source_lang):
            raise ValueError(
                f"Invalid source language: {source_lang}. "
                f"Use /languages endpoint to see supported languages."
            )

        if not is_valid_language(target_lang):
            raise ValueError(
                f"Invalid target language: {target_lang}. "
                f"Use /languages endpoint to see supported languages."
            )

        logger.debug(
            f"Languages validated: {get_language_name(source_lang)} -> "
            f"{get_language_name(target_lang)}"
        )

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """
        Translate text from source to target language.

        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            Translated text

        Raises:
            RuntimeError: If model is not loaded
            ValueError: If language codes are invalid
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Validate languages
        self.validate_languages(source_lang, target_lang)

        try:
            # Set source language
            self.tokenizer.src_lang = source_lang

            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=settings.MAX_LENGTH,
            )

            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get target language token ID
            # Handle both old and new tokenizer API
            if hasattr(self.tokenizer, 'lang_code_to_id'):
                target_lang_id = self.tokenizer.lang_code_to_id[target_lang]
            else:
                # For newer transformers versions, convert token to ID
                target_lang_id = self.tokenizer.convert_tokens_to_ids(target_lang)

            # Generate translation with no gradient calculation
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=target_lang_id,
                    max_length=settings.MAX_LENGTH,
                    num_beams=5,
                    early_stopping=True,
                )

            # Decode output
            translated_text = self.tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )[0]

            logger.debug(f"Translation completed: {len(text)} -> {len(translated_text)} chars")

            return translated_text

        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            raise RuntimeError(f"Translation error: {str(e)}")

    def is_gpu_available(self) -> bool:
        """Check if GPU is available and being used."""
        return self.device == "cuda"

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_name": settings.MODEL_NAME,
            "custom_model": settings.CUSTOM_MODEL_PATH is not None,
            "device": self.device,
            "half_precision": settings.USE_HALF_PRECISION and self.device == "cuda",
            "model_loaded": self.model_loaded,
        }


# Singleton instance
_translation_service: Optional[TranslationService] = None


@lru_cache()
def get_translation_service() -> TranslationService:
    """Get or create translation service singleton."""
    global _translation_service
    if _translation_service is None:
        _translation_service = TranslationService()
    return _translation_service


