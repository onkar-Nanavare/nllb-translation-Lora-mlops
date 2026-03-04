import os
import sys
import argparse
import yaml
import json
from pathlib import Path
from typing import Dict, Optional
import logging
import re

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset as HFDataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from training.train import TranslationDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --------------------------------------------------
# Utility Functions
# --------------------------------------------------

def load_training_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_lora_config(config: Dict) -> LoraConfig:
    lora_params = config["lora"]
    return LoraConfig(
        r=lora_params["r"],
        lora_alpha=lora_params["lora_alpha"],
        target_modules=lora_params["target_modules"],
        lora_dropout=lora_params["lora_dropout"],
        bias=lora_params["bias"],
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
    )


def get_latest_checkpoint(output_dir: str) -> Optional[str]:
    if not os.path.exists(output_dir):
        return None

    checkpoints = []
    for folder in os.listdir(output_dir):
        if folder.startswith("checkpoint-"):
            step = int(re.findall(r"\d+", folder)[0])
            checkpoints.append((step, folder))

    if not checkpoints:
        return None

    latest = sorted(checkpoints, key=lambda x: x[0])[-1][1]
    return os.path.join(output_dir, latest)


def load_glossary_pairs(glossary_path: str):
    with open(glossary_path, "r", encoding="utf-8") as f:
        glossary = json.load(f)
    pairs = []
    for domain in glossary.values():
        for src, tgt in domain.items():
            pairs.append((src, tgt))
    return pairs


def preprocess_function(examples, tokenizer, max_length):
    inputs = tokenizer(
        examples["source"],
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    labels = tokenizer(
        examples["target"],
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )

    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]

    inputs["labels"] = labels["input_ids"]
    return inputs


# --------------------------------------------------
# Main Training Logic
# --------------------------------------------------

def train_lora(
    data_file: str,
    glossary_file: str,
    source_lang: str,
    target_lang: str,
    output_dir: str,
    model_name: str,
    config_path: str,
):

    logger.info("🚀 Starting NLLB LoRA fine-tuning pipeline")
    config = load_training_config(config_path)

    # Load dataset
    dataset = TranslationDataset(data_file, source_lang, target_lang)
    hf_dataset = dataset.to_hf_dataset()

    glossary_pairs = load_glossary_pairs(glossary_file)
    glossary_dataset = HFDataset.from_dict({
        "source": [p[0] for p in glossary_pairs],
        "target": [p[1] for p in glossary_pairs],
    })

    combined_sources = list(hf_dataset["source"]) + list(glossary_dataset["source"]) * 2
    combined_targets = list(hf_dataset["target"]) + list(glossary_dataset["target"]) * 2

    hf_dataset = HFDataset.from_dict({
        "source": combined_sources,
        "target": combined_targets,
    }).shuffle(seed=config["seed"])

    split = hf_dataset.train_test_split(
        test_size=config["training"]["data"]["eval_split"]
    )

    train_dataset = split["train"]
    eval_dataset = split["test"]

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.src_lang = source_lang

    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    base_model.config.use_cache = False

    latest_checkpoint = get_latest_checkpoint(output_dir)

    # --------------------------------------------------
    # Resume Logic
    # --------------------------------------------------

    if latest_checkpoint:
        logger.info(f"🔁 Found checkpoint: {latest_checkpoint}")
        model = PeftModel.from_pretrained(base_model, latest_checkpoint, is_trainable=True)
        resume_checkpoint = latest_checkpoint

    elif os.path.exists(output_dir) and os.path.exists(
        os.path.join(output_dir, "adapter_model.safetensors")
    ):
        logger.info("🔁 No checkpoint found. Loading existing adapter weights.")
        model = PeftModel.from_pretrained(base_model, output_dir, is_trainable=True)
        resume_checkpoint = None

    else:
        logger.info("🆕 No previous model found. Starting fresh LoRA training.")
        lora_config = create_lora_config(config)
        model = get_peft_model(base_model, lora_config)
        resume_checkpoint = None

    model.train()
    model.print_trainable_parameters()

    if hasattr(tokenizer, "lang_code_to_id"):
        model.config.forced_bos_token_id = tokenizer.lang_code_to_id[target_lang]

    max_length = config["preprocessing"]["max_source_length"]

    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    eval_dataset = eval_dataset.map(
        lambda x: preprocess_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=eval_dataset.column_names,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    hp = config["training"]["hyperparameters"]

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=hp["per_device_train_batch_size"],
        per_device_eval_batch_size=hp["per_device_eval_batch_size"],
        gradient_accumulation_steps=hp["gradient_accumulation_steps"],
        learning_rate=hp["learning_rate"],
        num_train_epochs=hp["num_train_epochs"],
        save_strategy=config["training"]["save_strategy"],
        save_steps=config["training"]["save_steps"],
        save_total_limit=config["training"]["save_total_limit"],
        evaluation_strategy=config["training"]["evaluation"]["strategy"],
        logging_strategy=config["training"]["logging"]["strategy"],
        logging_steps=config["training"]["logging"]["steps"],
        fp16=device == "cuda",
        optim="adamw_torch",
        report_to="none",
        load_best_model_at_end=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    if resume_checkpoint:
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info(f"✅ Training complete. Model saved to {output_dir}")


# --------------------------------------------------
# CLI
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--glossary-file", required=True)
    parser.add_argument("--source-lang", default="eng_Latn")
    parser.add_argument("--target-lang", default="hin_Deva")
    parser.add_argument("--output-dir", default="./models/custom-nllb")
    parser.add_argument("--model-name", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--config", default="configs/training_config.yaml")

    args = parser.parse_args()

    train_lora(
        data_file=args.data_file,
        glossary_file=args.glossary_file,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        output_dir=args.output_dir,
        model_name=args.model_name,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
