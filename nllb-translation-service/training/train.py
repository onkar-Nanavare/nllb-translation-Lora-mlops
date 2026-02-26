"""
Training script for fine-tuning NLLB model on custom domain data.
"""
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import logging

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset as HFDataset

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TranslationDataset:
    """Custom dataset for translation training."""

    def __init__(
        self,
        data_file: str,
        source_lang: str,
        target_lang: str,
        delimiter: str = "\t"
    ):
        """
        Initialize translation dataset.

        Args:
            data_file: Path to TSV file with source_text<TAB>target_text
            source_lang: Source language code
            target_lang: Target language code
            delimiter: Delimiter in the data file (default: tab)
        """
        self.data_file = data_file
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.delimiter = delimiter
        self.examples = self._load_data()

    def _load_data(self) -> List[Tuple[str, str]]:
        """Load data from TSV file."""
        examples = []

        logger.info(f"Loading data from {self.data_file}")

        with open(self.data_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split(self.delimiter)
                if len(parts) != 2:
                    logger.warning(
                        f"Skipping line {line_num}: expected 2 columns, got {len(parts)}"
                    )
                    continue

                source_text, target_text = parts
                examples.append((source_text.strip(), target_text.strip()))

        logger.info(f"Loaded {len(examples)} examples")
        return examples

    def to_hf_dataset(self) -> HFDataset:
        """Convert to HuggingFace Dataset."""
        data = {
            "source": [ex[0] for ex in self.examples],
            "target": [ex[1] for ex in self.examples],
        }
        return HFDataset.from_dict(data)


def preprocess_function(
    examples: Dict,
    tokenizer: AutoTokenizer,
    source_lang: str,
    target_lang: str,
    max_length: int = 512,
):
    """
    Preprocess examples for training.

    Args:
        examples: Batch of examples
        tokenizer: Tokenizer instance
        source_lang: Source language code
        target_lang: Target language code
        max_length: Maximum sequence length

    Returns:
        Processed model inputs
    """
    # Set source language
    tokenizer.src_lang = source_lang

    # Tokenize inputs
    inputs = tokenizer(
        examples["source"],
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )

    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["target"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

    inputs["labels"] = labels["input_ids"]
    return inputs


def train(
    data_file: str,
    source_lang: str,
    target_lang: str,
    output_dir: str,
    model_name: str = "facebook/nllb-200-distilled-600M",
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    eval_split: float = 0.1,
    max_length: int = 512,
) -> None:
    """
    Train NLLB model on custom data.

    Args:
        data_file: Path to training data TSV file
        source_lang: Source language code
        target_lang: Target language code
        output_dir: Directory to save fine-tuned model
        model_name: Base model name
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        eval_split: Fraction of data to use for evaluation
        max_length: Maximum sequence length
    """
    logger.info("=" * 80)
    logger.info("Starting NLLB Fine-tuning")
    logger.info("=" * 80)
    logger.info(f"Model: {model_name}")
    logger.info(f"Source Language: {source_lang}")
    logger.info(f"Target Language: {target_lang}")
    logger.info(f"Data File: {data_file}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info("=" * 80)

    # Load dataset
    dataset = TranslationDataset(
        data_file=data_file,
        source_lang=source_lang,
        target_lang=target_lang,
    )
    hf_dataset = dataset.to_hf_dataset()

    # Split dataset
    if eval_split > 0:
        split_dataset = hf_dataset.train_test_split(test_size=eval_split)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        logger.info(f"Train examples: {len(train_dataset)}")
        logger.info(f"Eval examples: {len(eval_dataset)}")
    else:
        train_dataset = hf_dataset
        eval_dataset = None
        logger.info(f"Train examples: {len(train_dataset)}")

    # Load tokenizer and model
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        src_lang=source_lang,
    )

    logger.info("Loading model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Preprocess datasets
    logger.info("Preprocessing data...")
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(
            x, tokenizer, source_lang, target_lang, max_length
        ),
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    if eval_dataset:
        eval_dataset = eval_dataset.map(
            lambda x: preprocess_function(
                x, tokenizer, source_lang, target_lang, max_length
            ),
            batched=True,
            remove_columns=eval_dataset.column_names,
        )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
    )

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch" if eval_dataset else "no",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=2,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("=" * 80)
    logger.info("Training complete!")
    logger.info("=" * 80)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Fine-tune NLLB model on custom domain data"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        required=True,
        help="Path to TSV file with source_text<TAB>target_text"
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
        "--output-dir",
        type=str,
        default="./models/custom-nllb",
        help="Output directory for fine-tuned model"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="facebook/nllb-200-distilled-600M",
        help="Base model name"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--eval-split",
        type=float,
        default=0.1,
        help="Fraction of data for evaluation (0 to disable)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )

    args = parser.parse_args()

    # Run training
    train(
        data_file=args.data_file,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        output_dir=args.output_dir,
        model_name=args.model_name,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_split=args.eval_split,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
