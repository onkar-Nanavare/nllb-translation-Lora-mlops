#!/usr/bin/env python3
"""
Script to pre-download NLLB model.
Useful for Docker builds or CI/CD pipelines.
"""
import argparse
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def download_model(model_name: str, cache_dir: str = "./models/cache") -> None:
    """
    Download NLLB model and tokenizer.

    Args:
        model_name: Model name or path
        cache_dir: Directory to cache model files
    """
    print("=" * 80)
    print(f"Downloading model: {model_name}")
    print(f"Cache directory: {cache_dir}")
    print("=" * 80)

    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)

    # Download tokenizer
    print("\nDownloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
    )
    print("✓ Tokenizer downloaded")

    # Download model
    print("\nDownloading model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
    )
    print("✓ Model downloaded")

    print("\n" + "=" * 80)
    print("Download complete!")
    print("=" * 80)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Download NLLB model for offline use"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="facebook/nllb-200-distilled-600M",
        help="Model name to download"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./models/cache",
        help="Cache directory"
    )

    args = parser.parse_args()

    download_model(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
