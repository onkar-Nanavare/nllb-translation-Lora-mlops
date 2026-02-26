"""
Export NLLB model to ONNX format for optimized inference.
"""
import os
import sys
import argparse
from pathlib import Path
import logging

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from optimum.onnxruntime import ORTModelForSeq2SeqLM

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def export_to_onnx(
    model_path: str,
    output_path: str,
    opset_version: int = 14,
) -> None:
    """
    Export model to ONNX format.

    Args:
        model_path: Path to PyTorch model
        output_path: Path to save ONNX model
        opset_version: ONNX opset version
    """
    logger.info("=" * 80)
    logger.info("Exporting model to ONNX format")
    logger.info("=" * 80)
    logger.info(f"Model path: {model_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"ONNX opset version: {opset_version}")
    logger.info("=" * 80)

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    try:
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Export using Optimum
        logger.info("Converting model to ONNX...")
        logger.info("This may take several minutes...")

        # Load and convert model
        model = ORTModelForSeq2SeqLM.from_pretrained(
            model_path,
            export=True,
        )

        # Save ONNX model
        logger.info(f"Saving ONNX model to {output_path}")
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        logger.info("=" * 80)
        logger.info("ONNX export complete!")
        logger.info("=" * 80)
        logger.info(f"Model saved to: {output_path}")
        logger.info("\nTo use the ONNX model:")
        logger.info("1. Update .env: CUSTOM_MODEL_PATH=/path/to/onnx/model")
        logger.info("2. Install: pip install optimum[onnxruntime-gpu]")
        logger.info("3. Restart the service")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"ONNX export failed: {str(e)}")
        raise


def validate_onnx_model(
    onnx_model_path: str,
    test_text: str = "Hello, how are you?",
    source_lang: str = "eng_Latn",
    target_lang: str = "hin_Deva",
) -> None:
    """
    Validate exported ONNX model.

    Args:
        onnx_model_path: Path to ONNX model
        test_text: Test text for validation
        source_lang: Source language code
        target_lang: Target language code
    """
    logger.info("Validating ONNX model...")

    try:
        # Load ONNX model
        tokenizer = AutoTokenizer.from_pretrained(onnx_model_path)
        model = ORTModelForSeq2SeqLM.from_pretrained(onnx_model_path)

        # Set source language
        tokenizer.src_lang = source_lang

        # Tokenize
        inputs = tokenizer(test_text, return_tensors="pt")

        # Generate
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang) if not hasattr(tokenizer, 'lang_code_to_id') else tokenizer.lang_code_to_id[target_lang],
            max_length=128,
        )

        # Decode
        translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        logger.info("=" * 80)
        logger.info("ONNX Model Validation")
        logger.info("=" * 80)
        logger.info(f"Input ({source_lang}): {test_text}")
        logger.info(f"Output ({target_lang}): {translation}")
        logger.info("=" * 80)
        logger.info("Validation successful!")

    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        raise


def main():
    """Main export function."""
    parser = argparse.ArgumentParser(
        description="Export NLLB model to ONNX format"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to PyTorch model directory"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save ONNX model"
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate exported model"
    )
    parser.add_argument(
        "--test-text",
        type=str,
        default="Hello, how are you?",
        help="Test text for validation"
    )
    parser.add_argument(
        "--source-lang",
        type=str,
        default="eng_Latn",
        help="Source language for validation"
    )
    parser.add_argument(
        "--target-lang",
        type=str,
        default="hin_Deva",
        help="Target language for validation"
    )

    args = parser.parse_args()

    # Export to ONNX
    export_to_onnx(
        model_path=args.model_path,
        output_path=args.output_path,
        opset_version=args.opset_version,
    )

    # Validate if requested
    if args.validate:
        validate_onnx_model(
            onnx_model_path=args.output_path,
            test_text=args.test_text,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
        )


if __name__ == "__main__":
    main()
