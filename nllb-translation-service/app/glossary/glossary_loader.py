import json
import os
from typing import Dict
from ..core import get_settings, get_logger

logger = get_logger(__name__)
settings = get_settings()

_GLOSSARY_CACHE: Dict[str, str] | None = None


def load_glossary_from_file() -> Dict[str, str]:
    global _GLOSSARY_CACHE

    if _GLOSSARY_CACHE is not None:
        return _GLOSSARY_CACHE

    glossary_path = settings.GLOSSARY_PATH

    if not glossary_path or not os.path.exists(glossary_path):
        logger.warning(f"Glossary file not found: {glossary_path}")
        _GLOSSARY_CACHE = {}
        return _GLOSSARY_CACHE

    try:
        with open(glossary_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError("Glossary JSON must be key:value pairs")

        _GLOSSARY_CACHE = data
        logger.info(f"Glossary loaded: {len(_GLOSSARY_CACHE)} terms")
        return _GLOSSARY_CACHE

    except Exception as e:
        logger.error(f"Failed to load glossary: {e}")
        _GLOSSARY_CACHE = {}
        return _GLOSSARY_CACHE
