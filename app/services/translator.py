"""
NLLB Translation Service.
Handles model loading, caching, downloading, and translation operations.
"""

import os
import tarfile
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Optional
from functools import lru_cache

from ..core import get_settings, get_logger
from .language_codes import is_valid_language, get_language_name

logger = get_logger(__name__)
settings = get_settings()


class TranslationService:
    def __init__(self):
        self.model: Optional[AutoModelForSeq2SeqLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.device: str = "cpu"
        self.model_loaded: bool = False

    def _download_and_extract_model(self):
        if not settings.MODEL_URI:
            return

        os.makedirs(settings.CUSTOM_MODEL_PATH, exist_ok=True)
        archive_path = os.path.join(settings.CUSTOM_MODEL_PATH, "model.tar.gz")

        logger.info(f"Downloading model from {settings.MODEL_URI}")
        r = requests.get(settings.MODEL_URI, stream=True)
        r.raise_for_status()

        with open(archive_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info("Extracting model archive...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(settings.CUSTOM_MODEL_PATH)

        os.remove(archive_path)
        logger.info("Model downloaded and extracted successfully")

    def _resolve_model_path(self) -> str:
        if settings.CUSTOM_MODEL_PATH and os.path.exists(settings.CUSTOM_MODEL_PATH):
            logger.info(f"Using local model at {settings.CUSTOM_MODEL_PATH}")
            return settings.CUSTOM_MODEL_PATH

        if settings.MODEL_URI:
            self._download_and_extract_model()
            return settings.CUSTOM_MODEL_PATH

        logger.info("Falling back to base model")
        return settings.MODEL_NAME

    def load_model(self) -> None:
        try:
            model_path = self._resolve_model_path()

            if settings.USE_GPU and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"

            os.makedirs(settings.MODEL_CACHE_DIR, exist_ok=True)

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                cache_dir=settings.MODEL_CACHE_DIR,
                src_lang="eng_Latn",
            )

            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                cache_dir=settings.MODEL_CACHE_DIR,
            ).to(self.device)

            if self.device == "cuda" and settings.USE_HALF_PRECISION:
                self.model.half()

            self.model.eval()
            self.model_loaded = True
            logger.info("Model loaded successfully")

        except Exception as e:
            logger.exception("Model loading failed")
            raise RuntimeError(f"Model loading failed: {e}")

    def validate_languages(self, source_lang: str, target_lang: str):
        if not is_valid_language(source_lang):
            raise ValueError(f"Invalid source language: {source_lang}")
        if not is_valid_language(target_lang):
            raise ValueError(f"Invalid target language: {target_lang}")

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")

        self.validate_languages(source_lang, target_lang)

        self.tokenizer.src_lang = source_lang
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=settings.MAX_LENGTH,
        ).to(self.device)

        if hasattr(self.tokenizer, "lang_code_to_id"):
            forced_id = self.tokenizer.lang_code_to_id[target_lang]
        else:
            forced_id = self.tokenizer.convert_tokens_to_ids(target_lang)

        with torch.no_grad():
            tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_id,
                max_length=settings.MAX_LENGTH,
                num_beams=5,
            )

        return self.tokenizer.decode(tokens[0], skip_special_tokens=True)


_translation_service: Optional[TranslationService] = None


@lru_cache()
def get_translation_service() -> TranslationService:
    global _translation_service
    if _translation_service is None:
        _translation_service = TranslationService()
    return _translation_service