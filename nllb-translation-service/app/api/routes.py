"""
API routes for the translation service.
"""
from fastapi import APIRouter, HTTPException, Request
from typing import List
import asyncio
from ..glossary.glossary_loader import load_glossary_from_file


from ..models import (
    TranslationRequest,
    TranslationResponse,
    BatchTranslationRequest,
    BatchTranslationResponse,
    HealthResponse,
    LanguagesResponse,
    LanguageInfo,
)
from ..services import (
    get_translation_service,
    get_supported_languages,
)
from ..glossary import get_glossary_processor
from ..core import get_settings, get_logger

logger = get_logger(__name__)
settings = get_settings()
router = APIRouter()


@router.post("/translate", response_model=TranslationResponse)
async def translate(
    request: TranslationRequest,
    http_request: Request,
) -> TranslationResponse:
    """
    Translate text from source to target language.

    Args:
        request: Translation request with text and language codes
        http_request: HTTP request object for metadata

    Returns:
        Translation response with translated text

    Raises:
        HTTPException: If translation fails
    """
    request_id = getattr(http_request.state, "request_id", "unknown")

    try:
        logger.info(
            f"Translation request",
            extra={
                "request_id": request_id,
                "source_lang": request.source_lang,
                "target_lang": request.target_lang,
                "text_length": len(request.text),
            }
        )

        # Get services
        translator = get_translation_service()
        glossary_processor = get_glossary_processor()

        # Perform translation
        translated_text = translator.translate(
            text=request.text,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
        )

        # # Apply glossary if provided
        # glossary_applied = False
        # if request.glossary:
        #     translated_text, glossary_applied = glossary_processor.apply_glossary(
        #         translated_text=translated_text,
        #         original_text=request.text,
        #         glossary=request.glossary,
        #     )

        # Apply glossary from JSON file
        # glossary_applied = False
        # glossary = load_glossary_from_file()

        # if glossary:
        #     translated_text, glossary_applied = glossary_processor.apply_glossary(
        #     translated_text=translated_text,
        #     original_text=request.text,
        #     glossary=glossary,
        # )

        

        # Step 1: Decide glossary source (STRICT rule)
        if request.glossary:
            glossary_to_use = request.glossary
        elif settings.APPLY_GLOSSARY_FROM_FILE:
            glossary_to_use = load_glossary_from_file()
        else:
            glossary_to_use = None

        # Step 2: Apply glossary only if we have one
        glossary_applied = False
        if glossary_to_use:
            translated_text, glossary_applied = glossary_processor.apply_glossary(
            translated_text=translated_text,
            original_text=request.text,
            glossary=glossary_to_use,
        )

        logger.info(
            f"Translation completed",
            extra={
                "request_id": request_id,
                "glossary_applied": glossary_applied,
            }
        )

        return TranslationResponse(
            translated_text=translated_text,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            glossary_applied=glossary_applied,
        )

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}", extra={"request_id": request_id})
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Translation error: {str(e)}", extra={"request_id": request_id})
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(
            f"Unexpected error: {str(e)}",
            extra={"request_id": request_id},
            exc_info=True
        )
        raise HTTPException(status_code=500, detail="Translation failed")


@router.post("/batch-translate", response_model=BatchTranslationResponse)
async def batch_translate(
    request: BatchTranslationRequest,
    http_request: Request,
) -> BatchTranslationResponse:
    """
    Translate multiple texts in batch.

    Args:
        request: Batch translation request
        http_request: HTTP request object for metadata

    Returns:
        Batch translation response with all translations

    Raises:
        HTTPException: If batch translation fails
    """
    request_id = getattr(http_request.state, "request_id", "unknown")

    try:
        logger.info(
            f"Batch translation request",
            extra={
                "request_id": request_id,
                "batch_size": len(request.items),
            }
        )

        # Get services
        translator = get_translation_service()
        glossary_processor = get_glossary_processor()

        # Process all translations
        translations: List[TranslationResponse] = []

        for item in request.items:
            # Perform translation
            translated_text = translator.translate(
                text=item.text,
                source_lang=item.source_lang,
                target_lang=item.target_lang,
            )

            # # Apply glossary if provided
            # glossary_applied = False
            # if request.glossary:
            #     translated_text, glossary_applied = glossary_processor.apply_glossary(
            #         translated_text=translated_text,
            #         original_text=item.text,
            #         glossary=request.glossary,
            #     )


            # Apply glossary from JSON file
            glossary_applied = False
            glossary = load_glossary_from_file()

            if glossary:
                translated_text, glossary_applied = glossary_processor.apply_glossary(
                translated_text=translated_text,
                original_text=item.text,
                glossary=glossary,
            )


            translations.append(
                TranslationResponse(
                    translated_text=translated_text,
                    source_lang=item.source_lang,
                    target_lang=item.target_lang,
                    glossary_applied=glossary_applied,
                )
            )

        logger.info(
            f"Batch translation completed",
            extra={
                "request_id": request_id,
                "total": len(translations),
            }
        )

        return BatchTranslationResponse(
            translations=translations,
            total=len(translations),
        )

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}", extra={"request_id": request_id})
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Translation error: {str(e)}", extra={"request_id": request_id})
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(
            f"Unexpected error: {str(e)}",
            extra={"request_id": request_id},
            exc_info=True
        )
        raise HTTPException(status_code=500, detail="Batch translation failed")


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Health check endpoint.

    Returns:
        Health status information
    """
    translator = get_translation_service()

    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        model_loaded=translator.model_loaded,
        gpu_available=translator.is_gpu_available(),
    )


@router.get("/languages", response_model=LanguagesResponse)
async def get_languages() -> LanguagesResponse:
    """
    Get list of supported language codes.

    Returns:
        List of supported languages with codes and names
    """
    languages = get_supported_languages()

    language_list = [
        LanguageInfo(code=code, name=name)
        for code, name in languages.items()
    ]

    return LanguagesResponse(
        languages=language_list,
        total=len(language_list),
    )
