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
import re
import string
import torch
from torch.utils.data import Dataset, Subset
from transformers import TrainerCallback

from .constants import (
    DEFAULT_BASE_MODEL,
    NEUCODEC_REPO,
    SAMPLE_RATE_16K,
    SAMPLE_RATE_24K,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_TOP_K,
    DEFAULT_REPETITION_PENALTY,
    MAX_REF_CODES,
    TOKEN_TEXT_START,
    TOKEN_TEXT_END,
    TOKEN_SPEECH_START,
    TOKEN_SPEECH_END,
    PUNCT_SET,
)
from .phonemize import clean_reference_phones

_ASR_MODEL_DEFAULT = (
    "/mnt/volume_d2wey28/models/sherpa/"
    "sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-v2-2026-02-05/model.onnx"
)
_ASR_TOKENS_DEFAULT = (
    "/mnt/volume_d2wey28/models/sherpa/"
    "sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-v2-2026-02-05/tokens.txt"
)


class CEREvalCallback(TrainerCallback):
    """
    Online ASR CER evaluation callback.
    Synthesizes held-out evaluation sentences, transcribes with sherpa-onnx
    OmniASR 300M CTC, and reports `eval_cer` directly into Trainer metrics.
    """
    def __init__(self, eval_samples, tokenizer, asr_model, asr_tokens, codec):
        self.eval_samples = eval_samples
        self.tokenizer = tokenizer
        self.asr_model = asr_model
        self.asr_tokens = asr_tokens
        self.codec = codec
        self.rec = None
        self.speech_end_id = tokenizer.convert_tokens_to_ids(TOKEN_SPEECH_END)

    def _init_rec(self):
        if self.rec is None and os.path.exists(self.asr_model):
            try:
                import sherpa_onnx
                self.rec = sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
                    model=self.asr_model,
                    tokens=self.asr_tokens,
                    num_threads=2,
                    decoding_method="greedy_search",
                )
            except Exception as e:
                print(f"Notice: Could not initialize sherpa-onnx ({e}). Falling back to loss-only eval.")

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if metrics is None or model is None:
            return
        self._init_rec()
        if self.rec is None:
            return

        import jiwer
        import torchaudio
        import numpy as np

        device = next(model.parameters()).device
        was_training = model.training
        model.eval()

        cers = []
        sample_logs = []

        with torch.no_grad():
            for item in self.eval_samples:
                chat = item["chat"]
                text = item["text"]
                inp = torch.tensor([self.tokenizer.encode(chat)], device=device)

                try:
                    out = model.generate(
                        inp,
                        eos_token_id=self.speech_end_id,
                        max_new_tokens=220,
                        do_sample=True,
                        temperature=DEFAULT_TEMPERATURE,
                        top_p=DEFAULT_TOP_P,
                        top_k=DEFAULT_TOP_K,
                        repetition_penalty=DEFAULT_REPETITION_PENALTY,
                        use_cache=True,
                    )
                    gen_ids = out[0, inp.shape[1]:].cpu().tolist()
                    gen_str = self.tokenizer.decode(gen_ids, add_special_tokens=False)
                    sp_ids = [int(n) for n in re.findall(r"<\|speech_(\d+)\|>", gen_str)]
                except Exception:
                    sp_ids = []

                if len(sp_ids) < 15:
                    cers.append(1.0)
                    continue

                try:
                    sp_t = torch.tensor(sp_ids, device=device).unsqueeze(0).unsqueeze(0)
                    wav_24k = self.codec.decode_code(sp_t)
                    if wav_24k.numel() < 4800:
                        cers.append(1.0)
                        continue
                    wav_24k = wav_24k.squeeze().cpu().float()
                    if wav_24k.ndim != 1 or len(wav_24k) < 4800:
                        cers.append(1.0)
                        continue

                    wav_16k = torchaudio.functional.resample(wav_24k, SAMPLE_RATE_24K, SAMPLE_RATE_16K).numpy()
                    if len(wav_16k) < 3200:
                        cers.append(1.0)
                        continue

                    st = self.rec.create_stream()
                    st.accept_waveform(SAMPLE_RATE_16K, np.asarray(wav_16k, dtype=np.float32))
                    self.rec.decode_stream(st)
                    hyp = (st.result.text or "").strip()
                except Exception as exc:
                    hyp = ""

                ref_c = re.sub(r"[^\w\s]", "", text).lower().strip()
                hyp_c = re.sub(r"[^\w\s]", "", hyp).lower().strip()
                cer = jiwer.cer(ref_c, hyp_c) if hyp_c else 1.0
                cers.append(cer)
                sample_logs.append((text[:40], hyp[:40], cer))

        if was_training:
            model.train()

        mean_cer = sum(cers) / len(cers) if cers else 1.0
        metrics["eval_cer"] = mean_cer
        loss_val = metrics.get("eval_loss", 0.0)
        metrics["eval_joint_score"] = loss_val + (mean_cer * 10.0)

        print(f"\n>>> [ASR Eval @ Step {state.global_step}] Mean CER: {mean_cer:.2%} (Loss: {metrics.get('eval_loss', 0):.4f}) <<<", flush=True)
        for t, h, c in sample_logs[:2]:
            print(f"    REF: {t}...\n    ASR: {h}... (CER: {c:.2%})", flush=True)
        print("", flush=True)


class JointPlateauEarlyStoppingCallback(TrainerCallback):
    """
    Early stops when BOTH eval_loss AND eval_cer have plateaued.
    An evaluation check is considered an improvement if EITHER:
      - eval_loss improves by >= min_loss_delta, OR
      - eval_cer improves by >= min_cer_delta.
    Training continues as long as at least one metric is still improving.
    Only when BOTH have stagnated for `patience` checks does training stop.
    """
    def __init__(self, patience: int = 8, min_loss_delta: float = 0.005, min_cer_delta: float = 0.005):
        self.patience = patience
        self.min_loss_delta = min_loss_delta
        self.min_cer_delta = min_cer_delta
        self.best_loss = float("inf")
        self.best_cer = float("inf")
        self.loss_stagnant_count = 0
        self.cer_stagnant_count = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return

        current_loss = metrics.get("eval_loss")
        current_cer = metrics.get("eval_cer")

        if current_loss is not None:
            if current_loss < self.best_loss - self.min_loss_delta:
                self.best_loss = current_loss
                self.loss_stagnant_count = 0
            else:
                self.loss_stagnant_count += 1

        if current_cer is not None:
            if current_cer < self.best_cer - self.min_cer_delta:
                self.best_cer = current_cer
                self.cer_stagnant_count = 0
            else:
                self.cer_stagnant_count += 1

        if self.loss_stagnant_count >= self.patience and self.cer_stagnant_count >= self.patience:
            print(
                f"\n[Joint Plateau Triggered @ Step {state.global_step}] Both eval_loss "
                f"(stagnant {self.loss_stagnant_count} checks, best: {self.best_loss:.4f}) and eval_cer "
                f"(stagnant {self.cer_stagnant_count} checks, best: {self.best_cer:.2%}) have plateaued. "
                f"Stopping training automatically.\n",
                flush=True,
            )
            control.should_training_stop = True


class TTSDataset(Dataset):
    """
    Dataset for Neu-TTS training.
    Supports both:
    1. Zero-ref mode (no_ref=True): direct text-to-speech mapping without reference audio.
       <|TEXT_PROMPT_START|>{ipa_phones}<|TEXT_PROMPT_END|><|SPEECH_GENERATION_START|>{target_codes}<|SPEECH_GENERATION_END|>
    2. In-context mode (no_ref=False): voice-cloning pairing.
    """
    def __init__(self, metadata_path: str, tokenizer, max_len: int = 4096, no_ref: bool = True):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.no_ref = no_ref
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
        # Filter samples that exceed max_len tokens
        self.samples = [
            s for s in self.samples
            if len(self.tokenizer.encode(self._render(s))) <= max_len
        ]
        print(f"Length filter: kept {len(self.samples)}/{before} within {max_len} tokens (no_ref={no_ref}).")

        # Reference-eligible pool (for when in-context mode is enabled)
        self._ref_indices = [
            i for i, s in enumerate(self.samples)
            if 100 <= len(s["codes"]) <= 350
        ]
        if not self._ref_indices:
            self._ref_indices = list(range(len(self.samples)))

    def _render(self, target, ref=None):
        tgt_str = "".join(f"<|speech_{c}|>" for c in target["codes"])
        if self.no_ref or ref is None:
            return (
                f"{TOKEN_TEXT_START}{target['phones'].strip()}{TOKEN_TEXT_END}"
                f"{TOKEN_SPEECH_START}{tgt_str}{TOKEN_SPEECH_END}"
            )
        ref_codes = ref["codes"][:MAX_REF_CODES]
        ref_phones = clean_reference_phones(ref["phones"])
        combined = ref_phones.strip() + " " + target["phones"].strip()
        ref_str = "".join(f"<|speech_{c}|>" for c in ref_codes)
        return (
            f"{TOKEN_TEXT_START}{combined}{TOKEN_TEXT_END}"
            f"{TOKEN_SPEECH_START}{ref_str}{tgt_str}{TOKEN_SPEECH_END}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        target = self.samples[idx]
        if self.no_ref:
            chat = self._render(target)
        else:
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
          metadata_file: str = "metadata_encoded.csv",
          no_ref: bool = True,
          full_sft: bool = False,
          epochs: int = 100, learning_rate: float = 5e-5,
          batch_size: int = 8, grad_accum: int = 4,
          eval_fraction: float = 0.05, early_stopping_patience: int = 12,
          eval_steps: int = 100, save_steps: int = 100, logging_steps: int = 20,
          lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05,
          max_len: int = 4096, seed: int = 42,
          eval_cer: bool = False, asr_model: str = _ASR_MODEL_DEFAULT, asr_tokens: str = _ASR_TOKENS_DEFAULT):

    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
    )
    from peft import LoraConfig, TaskType, get_peft_model

    random.seed(seed)
    torch.manual_seed(seed)

    data_csv = os.path.join(data_dir, metadata_file)
    if not os.path.exists(data_csv):
        raise FileNotFoundError(f"{data_csv} not found. Run prep & encode first.")

    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"Loading tokenizer from: {base_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    full_dataset = TTSDataset(data_csv, tokenizer, max_len=max_len, no_ref=no_ref)

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
            self.no_ref = parent.no_ref
            self.samples = parent.samples
            self._ref_indices = parent._ref_indices
            self._eval_idxs = idxs
            self._rng = random.Random(1234)

        def __len__(self):
            return len(self._eval_idxs)

        def __getitem__(self, i):
            real_idx = self._eval_idxs[i]
            target = self.samples[real_idx]
            if self.no_ref:
                chat = self._render(target)
            else:
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

    if full_sft:
        print("Training mode: Full Parameter SFT (100% of model parameters trainable).")
        for p in model.parameters():
            p.requires_grad = True
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters for Full SFT: {n_trainable:,} (100%)")
    else:
        print("Training mode: LoRA adaptation.")
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
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

    has_asr = os.path.exists(asr_model) and os.path.exists(asr_tokens)
    metric_best = "eval_joint_score" if (eval_cer and has_asr) else "eval_loss"
    print(f"Optimization metric for best model & early stopping: {metric_best}")

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
        tf32=True,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        eval_strategy="steps" if eval_ds is not None else "no",
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=eval_ds is not None,
        metric_for_best_model=metric_best,
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        seed=seed,
    )

    callbacks = []
    if eval_ds is not None:
        if eval_cer and has_asr:
            from neucodec import NeuCodec
            codec = NeuCodec.from_pretrained(NEUCODEC_REPO).to("cuda" if torch.cuda.is_available() else "cpu").eval()

            from .phonemize import phonemize_text

            CANONICAL_TESTS = {
                "twi": [
                    "Onyankopɔn yɛ ɔdɔ na ɔdɔ obiara wɔ wiase.",
                    "Me kɔ sukuu da biara na mepɛ sɛ mesua ade.",
                    "Wo ho te sɛn? Me ho yɛ, medaase pa ara.",
                ],
                "fat": [
                    "Nyame yɛ kɛse na ɔwɔ tumi nyinaa.",
                    "Me ma wo akye, wo ho te dɛn ndɛ?",
                ],
                "gaa": [
                    "Nyɔŋmɔ ji mi nɔkwɛlɔ, nɔ ko nɔ ko ehiaŋ mi.",
                    "Te oyoo tɛŋŋ? Miye jogbaŋŋ, oyiwaladɔŋŋ.",
                ],
                "ada": [
                    "Mawu ji nɔkwɛlɔ kpakpa nɔ ko be ni maa hia mi.",
                ],
                "ewe": [
                    "Yehowa nye nye kplɔla, naneke mehiãam o.",
                    "Ndi na wò, èfɔ nyuiea? Mefɔ nyuie, akpe.",
                ],
                "dag": [
                    "Naawuni n-nyɛ n gubi, shɛli bi kpaŋsi ma.",
                    "Dasiba, a biem be wula? N biem nyɛla viɛnyɛla.",
                ],
                "eng": [
                    "Welcome to Ghana, the gateway to Africa.",
                    "Good morning, how are you doing today?",
                ],
            }

            # Detect language code
            first_fn = full_dataset.samples[0]["filename"].lower()
            detected_lang = "twi"
            if any(k in first_fn for k in ["asante", "akuapem", "twi"]):
                detected_lang = "twi"
            elif any(k in first_fn for k in ["gaa", "ga"]):
                detected_lang = "gaa"
            elif "ewe" in first_fn:
                detected_lang = "ewe"
            elif any(k in first_fn for k in ["fat", "fante"]):
                detected_lang = "fat"
            elif "dag" in first_fn:
                detected_lang = "dag"
            elif any(k in first_fn for k in ["dga", "dagaare"]):
                detected_lang = "dga"
            elif any(k in first_fn for k in ["gur", "frafra"]):
                detected_lang = "gur"
            elif any(k in first_fn for k in ["nzi", "nzema"]):
                detected_lang = "nzi"
            elif any(k in first_fn for k in ["aha", "ahanta"]):
                detected_lang = "aha"
            elif any(k in first_fn for k in ["sfw", "sehwi"]):
                detected_lang = "sfw"
            elif any(k in first_fn for k in ["ada", "dangme"]):
                detected_lang = "ada"
            elif any(k in first_fn for k in ["eng", "en"]):
                detected_lang = "eng"
            else:
                detected_lang = "twi"

            test_texts = CANONICAL_TESTS.get(detected_lang, [
                "Onyankopɔn yɛ ɔdɔ na ɔdɔ obiara wɔ wiase.",
                "Me kɔ sukuu da biara na mepɛ sɛ mesua ade.",
            ])

            eval_samples = []
            from africa_g2p import AfricaPipeline
            pipe = AfricaPipeline(lang=detected_lang, output="ipa")

            for txt in test_texts:
                if detected_lang in ("eng", "en", "english"):
                    import re
                    tokens = re.findall(r"[\w']+|[.,!?;:]", txt)
                    res = []
                    for tok in tokens:
                        if tok in ".,!?;:":
                            if res:
                                res[-1] = res[-1] + tok
                        else:
                            res.append(pipe.run(tok, sep="").strip())
                    ipa_words = " ".join(res)
                else:
                    ipa_words = pipe.run(txt.strip(), sep="").strip()

                if no_ref:
                    chat = f"{TOKEN_TEXT_START}{ipa_words.strip()}{TOKEN_TEXT_END}{TOKEN_SPEECH_START}"
                else:
                    ref_cand = full_dataset.samples[full_dataset._ref_indices[0]]
                    ref_phones = clean_reference_phones(ref_cand["phones"])
                    ref_codes = ref_cand["codes"][:MAX_REF_CODES]
                    ref_codes_str = "".join(f"<|speech_{c}|>" for c in ref_codes)
                    combined = ref_phones.strip() + " " + ipa_words.strip()
                    chat = f"{TOKEN_TEXT_START}{combined}{TOKEN_TEXT_END}{TOKEN_SPEECH_START}{ref_codes_str}"

                eval_samples.append({"chat": chat, "text": txt})

            callbacks.append(CEREvalCallback(eval_samples, tokenizer, asr_model, asr_tokens, codec))
            callbacks.append(JointPlateauEarlyStoppingCallback(
                patience=early_stopping_patience,
                min_loss_delta=0.005,
                min_cer_delta=0.005,
            ))
            print(f"Registered online ASR CER evaluator ({len(eval_samples)} {detected_lang} validation sentences, no_ref={no_ref}).")
        else:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))
            print(f"Registered EarlyStoppingCallback on eval_loss (patience={early_stopping_patience}).")

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
    print(f"Saving best model LoRA adapter (tracked by {metric_best}) -> {final_path}")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print("Training completed successfully.")


def main():
    ap = argparse.ArgumentParser(description="Finetuning for Neu-TTS Ghanaian models")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--metadata", default="metadata_encoded.csv")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--base-model", default="pnnbao-ump/VieNeu-TTS-0.3B")
    ap.add_argument("--full-sft", action="store_true", default=False,
                    help="Full parameter SFT (100% of weights trainable, no LoRA)")
    ap.add_argument("--no-ref", action="store_true", default=True,
                    help="Direct zero-ref text-to-speech training without reference audio")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--learning-rate", type=float, default=3e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--eval-fraction", type=float, default=0.05)
    ap.add_argument("--early-stopping-patience", type=int, default=10)
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--logging-steps", type=int, default=20)
    ap.add_argument("--eval-cer", action="store_true", default=True,
                    help="Evaluate ASR CER directly during training and select best checkpoint on CER")
    ap.add_argument("--asr-model", default=_ASR_MODEL_DEFAULT)
    ap.add_argument("--asr-tokens", default=_ASR_TOKENS_DEFAULT)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train(
        data_dir=args.data_dir,
        metadata_file=args.metadata,
        output_dir=args.output_dir,
        run_name=args.run_name,
        base_model=args.base_model,
        full_sft=args.full_sft,
        no_ref=args.no_ref,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        eval_fraction=args.eval_fraction,
        early_stopping_patience=args.early_stopping_patience,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        eval_cer=args.eval_cer,
        asr_model=args.asr_model,
        asr_tokens=args.asr_tokens,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_len=args.max_len,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
