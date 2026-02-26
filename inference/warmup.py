"""
Model warmup utilities to reduce cold start latency.
"""
import time
import logging
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)


DEFAULT_WARMUP_SAMPLES = [
    "Hello, how are you?",
    "This is a test translation.",
    "Machine learning is transforming the world.",
    "Welcome to our translation service.",
    "Thank you for using our API.",
]


def warm_up_model(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    device: str = "cuda",
    source_lang: str = "eng_Latn",
    target_lang: str = "hin_Deva",
    num_samples: int = 5,
    custom_samples: Optional[List[str]] = None,
    max_length: int = 512,
) -> None:
    """
    Warm up the model by running inference on sample texts.
    This helps reduce latency for the first real translation request.

    Args:
        model: Translation model
        tokenizer: Model tokenizer
        device: Device to use (cuda/cpu/mps)
        source_lang: Source language code
        target_lang: Target language code
        num_samples: Number of warmup samples
        custom_samples: Custom warmup samples (uses defaults if None)
        max_length: Maximum sequence length
    """
    logger.info("Starting model warmup...")
    start_time = time.time()

    # Use custom samples or defaults
    samples = custom_samples or DEFAULT_WARMUP_SAMPLES[:num_samples]

    # Set source language
    tokenizer.src_lang = source_lang

    try:
        # Ensure model is in eval mode
        model.eval()

        # Warmup with individual samples
        for i, text in enumerate(samples, 1):
            logger.debug(f"Warmup sample {i}/{len(samples)}: {text[:50]}...")

            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )

            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                _ = model.generate(
                    **inputs,
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang) if not hasattr(tokenizer, 'lang_code_to_id') else tokenizer.lang_code_to_id[target_lang],
                    max_length=max_length,
                    num_beams=5,
                    early_stopping=True,
                )

        # Warmup with a batch
        if len(samples) > 1:
            logger.debug(f"Warmup batch: {len(samples)} samples")

            inputs = tokenizer(
                samples,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                _ = model.generate(
                    **inputs,
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang) if not hasattr(tokenizer, 'lang_code_to_id') else tokenizer.lang_code_to_id[target_lang],
                    max_length=max_length,
                    num_beams=5,
                    early_stopping=True,
                )

        elapsed_time = time.time() - start_time
        logger.info(f"Model warmup complete in {elapsed_time:.2f}s")

    except Exception as e:
        logger.error(f"Model warmup failed: {str(e)}")
        # Don't raise - warmup failure shouldn't prevent service startup


def warm_up_with_profiling(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    device: str = "cuda",
    source_lang: str = "eng_Latn",
    target_lang: str = "hin_Deva",
) -> dict:
    """
    Warm up model and profile performance.

    Args:
        model: Translation model
        tokenizer: Model tokenizer
        device: Device to use
        source_lang: Source language code
        target_lang: Target language code

    Returns:
        Dictionary with profiling stats
    """
    logger.info("Starting model warmup with profiling...")

    stats = {
        "device": device,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "samples": [],
    }

    # Test samples of different lengths
    test_samples = [
        "Hello",  # Very short
        "This is a test translation.",  # Short
        "Machine learning and artificial intelligence are transforming the world.",  # Medium
        " ".join(["This is a longer test sentence."] * 5),  # Long
    ]

    tokenizer.src_lang = source_lang
    model.eval()

    for text in test_samples:
        start_time = time.time()

        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang) if not hasattr(tokenizer, 'lang_code_to_id') else tokenizer.lang_code_to_id[target_lang],
                max_length=512,
                num_beams=5,
                early_stopping=True,
            )

        translation = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

        elapsed_time = (time.time() - start_time) * 1000  # Convert to ms

        sample_stats = {
            "input_length": len(text),
            "input_tokens": inputs["input_ids"].shape[1],
            "output_length": len(translation),
            "output_tokens": generated.shape[1],
            "time_ms": round(elapsed_time, 2),
        }

        stats["samples"].append(sample_stats)

        logger.info(
            f"Sample: {len(text)} chars, {inputs['input_ids'].shape[1]} tokens -> "
            f"{len(translation)} chars, {generated.shape[1]} tokens in {elapsed_time:.2f}ms"
        )

    # Calculate averages
    avg_time = sum(s["time_ms"] for s in stats["samples"]) / len(stats["samples"])
    stats["avg_time_ms"] = round(avg_time, 2)

    logger.info(f"Model warmup and profiling complete. Avg time: {avg_time:.2f}ms")

    return stats
