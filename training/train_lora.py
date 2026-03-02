"""
LoRA/PEFT-based fine-tuning for NLLB model.
Medical domain EN->HI with glossary bias.
Production-ready MLOps training pipeline.
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
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset as HFDataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from training.train import TranslationDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def train_lora(
    data_file: str,
    glossary_file: str,
    source_lang: str,
    target_lang: str,
    output_dir: str,
    model_name: str,
    config_path: str,
    resume_from: Optional[str] = None,
):
    logger.info("🚀 Starting NLLB LoRA fine-tuning pipeline")

    config = load_training_config(config_path)

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

    split = hf_dataset.train_test_split(test_size=config["training"]["data"]["eval_split"])
    train_dataset = split["train"]
    eval_dataset = split["test"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.src_lang = source_lang

    base_model_path = model_name
    finetuned_path = config.get("model", {}).get("finetuned_path")

    if resume_from and os.path.exists(resume_from):
        logger.info(f"🔁 Resuming from LoRA adapter: {resume_from}")
    elif finetuned_path and os.path.exists(finetuned_path):
        resume_from = finetuned_path
        logger.info(f"🔁 Auto-resuming from latest adapter: {finetuned_path}")

    logger.info(f"📦 Loading base model: {base_model_path}")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=None,  # ✅ Windows-safe
    )

    model.config.use_cache = False  # ✅ required for gradient checkpointing

    if resume_from and os.path.exists(resume_from):
        model = PeftModel.from_pretrained(model, resume_from, is_trainable=True)
    else:
        lora_config = create_lora_config(config)
        model = get_peft_model(model, lora_config)

    model.gradient_checkpointing_enable(use_reentrant=False)
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

        evaluation_strategy=eval_cfg["strategy"],

        fp16=torch.cuda.is_available(),
        gradient_checkpointing=opt["gradient_checkpointing"],

        optim="adamw_torch",
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

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info(f"✅ LoRA adapter saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--glossary-file", required=True)
    parser.add_argument("--source-lang", default="eng_Latn")
    parser.add_argument("--target-lang", default="hin_Deva")
    parser.add_argument("--output-dir", default="./models/custom-nllb/latest")
    parser.add_argument("--model-name", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--resume-from", default=None)

    args = parser.parse_args()

    train_lora(
        data_file=args.data_file,
        glossary_file=args.glossary_file,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        output_dir=args.output_dir,
        model_name=args.model_name,
        config_path=args.config,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
