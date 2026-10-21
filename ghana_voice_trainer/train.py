"""
LoRA Finetuning for Neu-TTS Ghanaian Speech Models.

Features:
- Dynamic per-batch sequence padding (~3x training speedup).
- In-context voice-cloning prompt format matching VieNeu-TTS.
- Random reference audio pairing from punctuation-free reference pool.
- Cosine LR scheduling with warmup and early stopping on validation loss.
- Targets attention & MLP projections (q, k, v, o, gate, up, down).
"""

import argparse
import json
import os
import random
import string
import torch
from torch.utils.data import Dataset, Subset

from .constants import (
    DEFAULT_BASE_MODEL,
    MAX_REF_CODES,
    TOKEN_TEXT_START,
    TOKEN_TEXT_END,
    TOKEN_SPEECH_START,
    TOKEN_SPEECH_END,
    PUNCT_SET,
)
from .phonemize import clean_reference_phones


class TTSDataset(Dataset):
    """
    Dataset of in-context voice-cloning pairs:
      <|TEXT_PROMPT_START|>{ref_phones} {target_phones}<|TEXT_PROMPT_END|>
      <|SPEECH_GENERATION_START|>{ref_codes}{target_codes}<|SPEECH_GENERATION_END|>
    """
    def __init__(self, metadata_path: str, tokenizer, max_len: int = 4096):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []

        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 3:
                    self.samples.append({
                        "filename": parts[0],
                        "text": parts[1],
                        "phones": parts[1],
                        "codes": json.loads(parts[2]),
                    })

        before = len(self.samples)
        # Filter samples that exceed max_len tokens when paired with self
        self.samples = [
            s for s in self.samples
            if len(self.tokenizer.encode(self._render(s, s))) <= max_len
        ]
        print(f"Length filter: kept {len(self.samples)}/{before} within {max_len} tokens.")

        # Reference-eligible pool: 2.0 - 6.0 seconds (100 - 300 codes)
        self._ref_indices = [
            i for i, s in enumerate(self.samples)
            if 100 <= len(s["codes"]) <= 350
        ]
        if not self._ref_indices:
            self._ref_indices = list(range(len(self.samples)))
        print(f"Reference pool: {len(self._ref_indices)} eligible reference samples.")

    def _render(self, target, ref):
        ref_codes = ref["codes"][:MAX_REF_CODES]
        ref_phones = clean_reference_phones(ref["phones"])
        combined = ref_phones.strip() + " " + target["phones"].strip()
        ref_str = "".join(f"<|speech_{c}|>" for c in ref_codes)
        tgt_str = "".join(f"<|speech_{c}|>" for c in target["codes"])
        return (
            f"{TOKEN_TEXT_START}{combined}{TOKEN_TEXT_END}"
            f"{TOKEN_SPEECH_START}{ref_str}{tgt_str}{TOKEN_SPEECH_END}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        target = self.samples[idx]
        candidates = [i for i in self._ref_indices if i != idx]
        if not candidates:
            candidates = [i for i in range(len(self.samples)) if i != idx]
        ref = self.samples[random.choice(candidates)]

        chat = self._render(target, ref)
        ids = self.tokenizer.encode(chat)
        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = torch.full_like(input_ids, -100)

        start_id = self.tokenizer.convert_tokens_to_ids(TOKEN_SPEECH_START)
        starts = (input_ids == start_id).nonzero(as_tuple=True)[0]
        if len(starts):
            labels[starts[0]:] = input_ids[starts[0]:]

        return {"input_ids": input_ids, "labels": labels}


def build_dynamic_collator(pad_token_id: int):
    """Dynamically pad each batch to its own longest item (avoids global 4096 waste)."""
    from torch.nn.utils.rnn import pad_sequence

    def collate(batch):
        ids = pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=pad_token_id)
        labels = pad_sequence([b["labels"] for b in batch], batch_first=True, padding_value=-100)
        mask = (ids != pad_token_id).long()
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}

    return collate


def train(data_dir: str, output_dir: str, run_name: str,
          base_model: str = DEFAULT_BASE_MODEL,
          epochs: int = 20, learning_rate: float = 5e-5,
          batch_size: int = 8, grad_accum: int = 4,
          eval_fraction: float = 0.05, early_stopping_patience: int = 4,
          eval_steps: int = 100, save_steps: int = 100, logging_steps: int = 20,
          lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05,
          max_len: int = 4096, seed: int = 42):

    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
    )
    from peft import LoraConfig, TaskType, get_peft_model

    random.seed(seed)
    torch.manual_seed(seed)

    data_csv = os.path.join(data_dir, "metadata_encoded.csv")
    if not os.path.exists(data_csv):
        raise FileNotFoundError(f"{data_csv} not found. Run prep & encode first.")

    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"Loading tokenizer from: {base_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    full_dataset = TTSDataset(data_csv, tokenizer, max_len=max_len)

    # Train / eval split
    n_eval = max(1, int(len(full_dataset.samples) * eval_fraction))
    indices = list(range(len(full_dataset.samples)))
    random.shuffle(indices)
    eval_indices = set(indices[:n_eval])
    train_indices = [i for i in indices[n_eval:]]

    train_ds = Subset(full_dataset, train_indices)

    # Stable eval dataset class
    class EvalDataset(TTSDataset):
        def __init__(self, parent, idxs):
            self.tokenizer = parent.tokenizer
            self.max_len = parent.max_len
            self.samples = parent.samples
            self._ref_indices = parent._ref_indices
            self._eval_idxs = idxs
            self._rng = random.Random(1234)

        def __len__(self):
            return len(self._eval_idxs)

        def __getitem__(self, i):
            real_idx = self._eval_idxs[i]
            target = self.samples[real_idx]
            candidates = [j for j in self._ref_indices if j != real_idx]
            if not candidates:
                candidates = [j for j in range(len(self.samples)) if j != real_idx]
            ref = self.samples[self._rng.choice(candidates)]

            chat = self._render(target, ref)
            ids = self.tokenizer.encode(chat)
            input_ids = torch.tensor(ids, dtype=torch.long)
            labels = torch.full_like(input_ids, -100)
            start_id = self.tokenizer.convert_tokens_to_ids(TOKEN_SPEECH_START)
            starts = (input_ids == start_id).nonzero(as_tuple=True)[0]
            if len(starts):
                labels[starts[0]:] = input_ids[starts[0]:]
            return {"input_ids": input_ids, "labels": labels}

    eval_ds = EvalDataset(full_dataset, sorted(eval_indices)) if eval_indices else None

    print(f"Loading base model: {base_model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    steps_per_epoch = max(1, len(train_ds) // (batch_size * grad_accum))
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(total_steps * 0.03))

    training_args = TrainingArguments(
        output_dir=run_dir,
        do_train=True,
        do_eval=eval_ds is not None,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        eval_strategy="steps" if eval_ds is not None else "no",
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=eval_ds is not None,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        seed=seed,
    )

    callbacks = []
    if eval_ds is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=build_dynamic_collator(tokenizer.pad_token_id),
        callbacks=callbacks,
    )

    print(f"Training on {len(train_ds)} samples (eval {len(eval_indices)}) up to {epochs} epochs ...")
    trainer.train()

    final_path = os.path.join(run_dir, "final")
    print(f"Saving LoRA adapter -> {final_path}")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print("Training completed successfully.")


def main():
    ap = argparse.ArgumentParser(description="LoRA Finetuning for Neu-TTS Ghanaian models")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--learning-rate", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--eval-fraction", type=float, default=0.05)
    ap.add_argument("--early-stopping-patience", type=int, default=4)
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--logging-steps", type=int, default=20)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        run_name=args.run_name,
        base_model=args.base_model,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        eval_fraction=args.eval_fraction,
        early_stopping_patience=args.early_stopping_patience,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_len=args.max_len,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
