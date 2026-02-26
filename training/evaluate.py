"""
Evaluation script for NLLB translation model.
Computes BLEU score and other metrics.
"""
import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict
import logging

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_metric
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluator for translation models."""

    def __init__(
        self,
        model_path: str,
        source_lang: str,
        target_lang: str,
        device: str = None
    ):
        """
        Initialize evaluator.

        Args:
            model_path: Path to model
            source_lang: Source language code
            target_lang: Target language code
            device: Device to use (cuda/cpu)
        """
        self.model_path = model_path
        self.source_lang = source_lang
        self.target_lang = target_lang

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")

        # Load model and tokenizer
        self._load_model()

    def _load_model(self) -> None:
        """Load model and tokenizer."""
        logger.info(f"Loading model from {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            src_lang=self.source_lang,
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

        logger.info("Model loaded successfully")

    def translate(self, text: str, max_length: int = 512) -> str:
        """
        Translate a single text.

        Args:
            text: Text to translate
            max_length: Maximum length

        Returns:
            Translated text
        """
        # Set source language
        self.tokenizer.src_lang = self.source_lang

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.target_lang) if not hasattr(self.tokenizer, 'lang_code_to_id') else self.tokenizer.lang_code_to_id[self.target_lang],
                max_length=max_length,
                num_beams=5,
                early_stopping=True,
            )

        # Decode
        translation = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )[0]

        return translation

    def evaluate(
        self,
        test_file: str,
        delimiter: str = "\t"
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.

        Args:
            test_file: Path to test file (source<TAB>target)
            delimiter: Delimiter in test file

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating on {test_file}")

        # Load test data
        sources = []
        references = []
        predictions = []

        with open(test_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split(delimiter)
                if len(parts) != 2:
                    logger.warning(
                        f"Skipping line {line_num}: expected 2 columns, got {len(parts)}"
                    )
                    continue

                source, reference = parts
                sources.append(source.strip())
                references.append(reference.strip())

        logger.info(f"Loaded {len(sources)} test examples")

        # Generate predictions
        logger.info("Generating translations...")
        for i, source in enumerate(sources, 1):
            if i % 10 == 0:
                logger.info(f"Translated {i}/{len(sources)}")

            prediction = self.translate(source)
            predictions.append(prediction)

        # Compute metrics
        logger.info("Computing metrics...")
        metrics = self._compute_metrics(predictions, references)

        return metrics

    def _compute_metrics(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.

        Args:
            predictions: List of predictions
            references: List of references

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # BLEU score
        try:
            bleu_metric = load_metric("sacrebleu")
            # sacrebleu expects references as list of lists
            refs_for_bleu = [[ref] for ref in references]
            bleu_result = bleu_metric.compute(
                predictions=predictions,
                references=refs_for_bleu
            )
            metrics["bleu"] = bleu_result["score"]
        except Exception as e:
            logger.warning(f"Failed to compute BLEU: {str(e)}")
            metrics["bleu"] = 0.0

        # Character-level accuracy
        char_matches = 0
        total_chars = 0
        for pred, ref in zip(predictions, references):
            for p_char, r_char in zip(pred, ref):
                if p_char == r_char:
                    char_matches += 1
            total_chars += max(len(pred), len(ref))

        metrics["char_accuracy"] = (char_matches / total_chars * 100) if total_chars > 0 else 0

        # Average length ratio
        length_ratios = [
            len(pred) / len(ref) if len(ref) > 0 else 0
            for pred, ref in zip(predictions, references)
        ]
        metrics["avg_length_ratio"] = np.mean(length_ratios)

        return metrics


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate NLLB translation model"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model directory"
    )
    parser.add_argument(
        "--test-file",
        type=str,
        required=True,
        help="Path to test file (source<TAB>target)"
    )
    parser.add_argument(
        "--source-lang",
        type=str,
        required=True,
        help="Source language code (e.g., eng_Latn)"
    )
    parser.add_argument(
        "--target-lang",
        type=str,
        required=True,
        help="Target language code (e.g., hin_Deva)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)"
    )

    args = parser.parse_args()

    # Create evaluator
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        device=args.device,
    )

    # Evaluate
    metrics = evaluator.evaluate(test_file=args.test_file)

    # Print results
    logger.info("=" * 80)
    logger.info("Evaluation Results")
    logger.info("=" * 80)
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value:.2f}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
