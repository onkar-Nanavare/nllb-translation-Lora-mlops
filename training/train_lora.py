

"""
LoRA/PEFT-based fine-tuning for NLLB model.
Medical domain EN->HI with glossary bias.
Optimized for small GPU (MX550, 2GB VRAM) and small dataset (2k examples).
"""

import os
import sys
import argparse
import yaml
import json
from pathlib import Path
from typing import Dict, Optional
import logging

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset as HFDataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from training.train import TranslationDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_training_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
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


def load_glossary_pairs(glossary_path: str):
    pairs = []
    with open(glossary_path, "r", encoding="utf-8") as f:
        glossary = json.load(f)
    for domain in glossary.values():
        for src, tgt in domain.items():
            pairs.append((src, tgt))
    return pairs


def preprocess_function(examples, tokenizer, max_length):
    inputs = examples["source"]
    targets = examples["target"]

    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )

    labels = tokenizer(
        targets,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )

    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def train_lora(
    data_file: str,
    glossary_file: str,
    source_lang: str,
    target_lang: str,
    output_dir: str,
    model_name: str,
    config_path: Optional[str] = None,
):
    logger.info("Starting NLLB LoRA fine-tuning...")

    config = load_training_config(config_path)

    # Load main dataset
    dataset = TranslationDataset(
        data_file=data_file,
        source_lang=source_lang,
        target_lang=target_lang,
    )
    hf_dataset = dataset.to_hf_dataset()

    # Load glossary and repeat for domain bias
    glossary_pairs = load_glossary_pairs(glossary_file)
    glossary_dataset = HFDataset.from_dict({
        "source": [p[0] for p in glossary_pairs],
        "target": [p[1] for p in glossary_pairs],
    })

    main_sources = list(hf_dataset["source"])
    main_targets = list(hf_dataset["target"])
    glossary_sources = list(glossary_dataset["source"])
    glossary_targets = list(glossary_dataset["target"])

    combined_sources = main_sources + glossary_sources * 2
    combined_targets = main_targets + glossary_targets * 2

    hf_dataset = HFDataset.from_dict({
        "source": combined_sources,
        "target": combined_targets,
    }).shuffle(seed=config["seed"])

    split = hf_dataset.train_test_split(test_size=config["training"]["data"]["eval_split"])
    train_dataset = split["train"]
    eval_dataset = split["test"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.src_lang = source_lang

    # Load model safely with GPU fallback
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.warning("GPU out of memory! Falling back to CPU...")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map={"": "cpu"}
            )

    # Force correct language token for NLLB
    if hasattr(tokenizer, "lang_code_to_id"):
        model.config.forced_bos_token_id = tokenizer.lang_code_to_id[target_lang]
    else:
        model.config.forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang)

    # LoRA setup
    lora_config = create_lora_config(config)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

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
    opt = config["training"]["optimization"]
    log_cfg = config["training"]["logging"]
    eval_cfg = config["training"]["evaluation"]

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=hp["per_device_train_batch_size"],
        per_device_eval_batch_size=hp["per_device_eval_batch_size"],
        gradient_accumulation_steps=hp["gradient_accumulation_steps"],
        learning_rate=hp["learning_rate"],
        num_train_epochs=hp["num_train_epochs"],
        save_strategy=config["training"]["save_strategy"],
        logging_strategy=log_cfg["strategy"],
        logging_steps=log_cfg["steps"],
        eval_strategy=eval_cfg["strategy"],
        fp16=opt["fp16"] if torch.cuda.is_available() else False,
        bf16=opt["bf16"] if torch.cuda.is_available() else False,
        gradient_checkpointing=opt["gradient_checkpointing"],
        optim=opt["optim"],
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    # Save final LoRA adapter
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"LoRA adapter saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--glossary-file", required=True)
    parser.add_argument("--source-lang", default="eng_Latn")
    parser.add_argument("--target-lang", default="hin_Deva")
    parser.add_argument("--output-dir", default="./models/lora-adapters/nllb-medical")
    parser.add_argument("--model-name", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--config", default="training_config.yaml")

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
