"""
Glossary processing module for term translation overrides.
"""
import re
from typing import Dict, Optional, Tuple
from ..core import get_logger

logger = get_logger(__name__)


class GlossaryProcessor:
    """Handles glossary-based term replacement in translations."""

    def __init__(self):
        """Initialize glossary processor."""
        pass

    def apply_glossary(
        self,
        translated_text: str,
        original_text: str,
        glossary: Dict[str, str],
        case_sensitive: bool = False,
    ) -> Tuple[str, bool]:
        """
        Apply glossary terms to translated text.

        Args:
            translated_text: The machine-translated text
            original_text: The original source text
            glossary: Dictionary of source term -> target term mappings
            case_sensitive: Whether to perform case-sensitive matching

        Returns:
            Tuple of (processed_text, applied) where applied indicates if any terms were replaced
        """
        if not glossary:
            return translated_text, False

        processed_text = translated_text
        applied = False

        # Track which terms were found in original text
        terms_to_apply = {}

        for source_term, target_term in glossary.items():
            # Check if the source term exists in the original text
            if case_sensitive:
                if source_term in original_text:
                    terms_to_apply[source_term] = target_term
            else:
                if source_term.lower() in original_text.lower():
                    terms_to_apply[source_term] = target_term

        if not terms_to_apply:
            logger.debug("No glossary terms found in original text")
            return translated_text, False

        # Apply replacements
        # Sort by length (longest first) to handle overlapping terms
        sorted_terms = sorted(
            terms_to_apply.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )

        for source_term, target_term in sorted_terms:
            # Try to find and replace the translated version of the term
            # This is a simple heuristic approach

            # For more sophisticated replacement, we could:
            # 1. Track word positions during translation
            # 2. Use alignment models
            # 3. Use fuzzy matching

            # Simple approach: Replace if the target term makes sense
            if target_term and len(target_term.strip()) > 0:
                # Create a word boundary regex pattern
                flags = 0 if case_sensitive else re.IGNORECASE

                # Try to preserve capitalization in simple cases
                # pattern = re.compile(
                #     r'\b' + re.escape(source_term) + r'\b',
                #     flags=flags
                # )

                pattern = re.compile(
                    re.escape(source_term), 
                    flags=flags
                )

                # Count matches before replacement
                matches_before = len(pattern.findall(processed_text))

                if matches_before > 0:
                    processed_text = pattern.sub(target_term, processed_text)
                    applied = True
                    logger.debug(
                        f"Applied glossary term: '{source_term}' -> '{target_term}' "
                        f"({matches_before} occurrences)"
                    )

        return processed_text, applied

    def validate_glossary(self, glossary: Dict[str, str]) -> bool:
        """
        Validate glossary format.

        Args:
            glossary: Glossary dictionary to validate

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(glossary, dict):
            return False

        for key, value in glossary.items():
            if not isinstance(key, str) or not isinstance(value, str):
                return False
            if len(key.strip()) == 0 or len(value.strip()) == 0:
                return False

        return True

    def merge_glossaries(
        self,
        *glossaries: Optional[Dict[str, str]]
    ) -> Dict[str, str]:
        """
        Merge multiple glossaries with later ones taking precedence.

        Args:
            *glossaries: Variable number of glossary dictionaries

        Returns:
            Merged glossary dictionary
        """
        merged = {}
        for glossary in glossaries:
            if glossary:
                merged.update(glossary)
        return merged


# Singleton instance
_glossary_processor: Optional[GlossaryProcessor] = None


def get_glossary_processor() -> GlossaryProcessor:
    """Get or create glossary processor singleton."""
    global _glossary_processor
    if _glossary_processor is None:
        _glossary_processor = GlossaryProcessor()
    return _glossary_processor
