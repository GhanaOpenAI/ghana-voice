"""
Merge a trained LoRA adapter into the base model to produce a standalone model.
"""

import argparse
import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .constants import DEFAULT_BASE_MODEL


def merge_lora(base_model: str, adapter_path: str, output_dir: str):
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading base model: {base_model} ...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    print(f"Loading LoRA adapter from: {adapter_path} ...")
    model = PeftModel.from_pretrained(base, adapter_path)

    print("Merging LoRA weights into base model ...")
    model = model.merge_and_unload()

    print(f"Saving merged standalone model -> {output_dir} ...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Merge complete!")


def main():
    ap = argparse.ArgumentParser(description="Merge LoRA adapter into base Neu-TTS model")
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--adapter-path", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    merge_lora(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
