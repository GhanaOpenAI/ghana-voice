"""
Main CLI entrypoint for ghana-voice.
"""

import argparse
import sys
from .constants import (
    get_language_info,
    TIER1_LANGUAGES,
    TIER2_LANGUAGES,
    DEFAULT_BASE_MODEL,
)


def run_pipeline(args):
    """All-in-one runner: prep -> encode -> train -> merge."""
    from .prep import prep_local_folder, prep_hf_dataset
    from .encode import encode_dataset
    from .train import train
    from .merge import merge_lora
    import os

    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True)
    lang_info = get_language_info(args.lang)

    print("\n" + "=" * 60)
    print(f"GHANA VOICE: {lang_info['display'].upper()}")
    print(f"Tier {lang_info['tier']} Language (Recommended Audio: {lang_info['rec_hours']})")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("STEP 1: PREPARING DATASET")
    print("=" * 60)
    if args.hf_repo:
        prep_hf_dataset(args.hf_repo, args.subset, args.split, data_dir, args.lang,
                        args.min_dur, args.max_dur, args.per_lang)
    elif args.input_dir:
        prep_local_folder(args.input_dir, args.metadata_file, data_dir, args.lang,
                          args.min_dur, args.max_dur, args.per_lang)
    else:
        print("Data already prepared, skipping prep step.")

    print("\n" + "=" * 60)
    print("STEP 2: ENCODING AUDIO WITH NEUCODEC")
    print("=" * 60)
    meta_encoded = os.path.join(data_dir, "metadata_encoded.csv")
    if os.path.exists(meta_encoded) and os.path.getsize(meta_encoded) > 0:
        print(f"{meta_encoded} already exists, skipping encoding.")
    else:
        encode_dataset(data_dir, batch_size=args.batch_size)

    print("\n" + "=" * 60)
    print("STEP 3: FINETUNING")
    print("=" * 60)
    train(
        data_dir=data_dir,
        output_dir=args.output_dir,
        run_name=args.run_name,
        base_model=args.base_model,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.train_batch_size,
        grad_accum=args.grad_accum,
        eval_fraction=args.eval_fraction,
        early_stopping_patience=args.early_stopping_patience,
    )

    print("\n" + "=" * 60)
    print("STEP 4: MERGING WEIGHTS INTO STANDALONE MODEL")
    print("=" * 60)
    adapter_path = os.path.join(args.output_dir, args.run_name, "final")
    merged_path = os.path.join(args.output_dir, args.run_name, "merged")
    merge_lora(
        base_model=args.base_model,
        adapter_path=adapter_path,
        output_dir=merged_path,
    )

    print("\n" + "=" * 60)
    print(f"PIPELINE COMPLETE! Merged model saved to: {merged_path}")
    print("=" * 60)


def print_languages_table(tier_filter=None):
    print("\n" + "=" * 70)
    print("GHANA VOICE: SUPPORTED LANGUAGES & TIER HIERARCHY")
    print("=" * 70)
    if tier_filter in (None, 1):
        print("\n🌟 TIER 1: Core Ghanaian Languages (Deeply tuned in Base Model)")
        print("   Recommended fine-tuning data: 2 to 3 hours (~1,000–1,500 clips)")
        print("-" * 70)
        print(f"   {'Identifier':16s} | {'Display Name':20s} | {'ISO':6s} | {'Family':12s}")
        print("-" * 70)
        for k, v in TIER1_LANGUAGES.items():
            print(f"   {k:16s} | {v['display']:20s} | {v['iso']:6s} | {v['family']:12s}")
    if tier_filter in (None, 2):
        print("\n🌍 TIER 2: Extended Ghanaian & Regional African Languages + Ghanaian English")
        print("   Recommended fine-tuning data: 3 to 6 hours (~1,000–3,000 clips)")
        print("-" * 70)
        print(f"   {'Identifier':16s} | {'Display Name':20s} | {'ISO':6s} | {'Family':12s}")
        print("-" * 70)
        for k, v in TIER2_LANGUAGES.items():
            print(f"   {k:16s} | {v['display']:20s} | {v['iso']:6s} | {v['family']:12s}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        prog="ghana-voice",
        description="Speech foundation models, inference & fine-tuning toolkit for Ghanaian & African languages",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: speak (CLI Inference)
    sub_speak = subparsers.add_parser("speak", help="Synthesize speech audio directly from text (Zero-Ref TTS)")
    sub_speak.add_argument("--text", required=True, help="Text to speak")
    sub_speak.add_argument("--lang", default="twi", help="Language identifier (e.g. twi, fat, ewe, gaa, eng, etc.)")
    sub_speak.add_argument("--model", default=None, help="Model path or HF repo ID (defaults to auto-resolving)")
    sub_speak.add_argument("--out", default="output.wav", help="Output WAV path")
    sub_speak.add_argument("--device", default=None, help="Device to run on ('cpu' or 'cuda')")
    sub_speak.add_argument("--ipa", action="store_true", help="Input is already IPA text")
    sub_speak.add_argument("--temperature", type=float, default=0.7)
    sub_speak.add_argument("--top-p", type=float, default=0.8)
    sub_speak.add_argument("--top-k", type=int, default=50)

    # Command: web (Web UI)
    sub_web = subparsers.add_parser("web", help="Launch interactive Gradio Web UI for voice synthesis")
    sub_web.add_argument("--port", type=int, default=7860, help="Web server port")
    sub_web.add_argument("--share", action="store_true", default=False, help="Create a public shareable link")

    # Command: serve (REST API)
    sub_serve = subparsers.add_parser("serve", help="Launch OpenAI-compatible REST API server")
    sub_serve.add_argument("--host", default="0.0.0.0", help="Host address")
    sub_serve.add_argument("--port", type=int, default=8000, help="Port")

    # Command: export-gguf (GGUF Export)
    sub_gguf = subparsers.add_parser("export-gguf", help="Export PyTorch model to GGUF format and quantize for CPU")
    sub_gguf.add_argument("--model-dir", required=True, help="Directory containing PyTorch model weights")
    sub_gguf.add_argument("--out-dir", required=True, help="Output directory for GGUF files")
    sub_gguf.add_argument("--quant", nargs="+", default=["Q4_K_M", "Q8_0"], help="Quantization types")

    # Command: list-langs
    sub_langs = subparsers.add_parser("list-langs", help="List all supported languages, tiers, and data requirements")
    sub_langs.add_argument("--tier", type=int, choices=[1, 2], default=None, help="Filter by Tier 1 or Tier 2")

    # Command: prep
    sub_prep = subparsers.add_parser("prep", help="Prepare audio and phonemes")
    sub_prep.add_argument("--out-dir", required=True)
    sub_prep.add_argument("--lang", required=True)
    sub_prep.add_argument("--input-dir", default="")
    sub_prep.add_argument("--metadata-file", default="metadata.csv")
    sub_prep.add_argument("--hf-repo", default="")
    sub_prep.add_argument("--subset", default="")
    sub_prep.add_argument("--split", default="train")
    sub_prep.add_argument("--min-dur", type=float, default=2.0)
    sub_prep.add_argument("--max-dur", type=float, default=15.0)
    sub_prep.add_argument("--per-lang", type=int, default=0)

    # Command: encode
    sub_enc = subparsers.add_parser("encode", help="NeuCodec encode prepared audio")
    sub_enc.add_argument("--data-dir", required=True)
    sub_enc.add_argument("--metadata", default="metadata.csv")
    sub_enc.add_argument("--output", default="metadata_encoded.csv")
    sub_enc.add_argument("--batch-size", type=int, default=16)

    # Command: train
    sub_train = subparsers.add_parser("train", help="Run fine-tuning (Full SFT or LoRA)")
    sub_train.add_argument("--data-dir", required=True)
    sub_train.add_argument("--metadata", default="metadata_encoded.csv")
    sub_train.add_argument("--output-dir", default="output")
    sub_train.add_argument("--run-name", required=True)
    sub_train.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    sub_train.add_argument("--full-sft", action="store_true", default=False, help="Train all 100%% of parameters (Full SFT, no LoRA)")
    sub_train.add_argument("--epochs", type=int, default=50)
    sub_train.add_argument("--learning-rate", type=float, default=3e-5)
    sub_train.add_argument("--batch-size", type=int, default=8)
    sub_train.add_argument("--grad-accum", type=int, default=4)
    sub_train.add_argument("--eval-fraction", type=float, default=0.05)
    sub_train.add_argument("--early-stopping-patience", type=int, default=10)
    sub_train.add_argument("--eval-steps", type=int, default=100)
    sub_train.add_argument("--save-steps", type=int, default=100)
    sub_train.add_argument("--logging-steps", type=int, default=20)
    sub_train.add_argument("--no-ref", action="store_true", default=True, help="Train zero-ref direct TTS without reference audio")
    sub_train.add_argument("--eval-cer", action="store_true", default=True)

    # Command: merge
    sub_merge = subparsers.add_parser("merge", help="Merge LoRA adapter into base model")
    sub_merge.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    sub_merge.add_argument("--adapter-path", required=True)
    sub_merge.add_argument("--output-dir", required=True)

    # Command: infer (In-context voice cloning)
    sub_infer = subparsers.add_parser("infer", help="Synthesize audio from text + reference audio")
    sub_infer.add_argument("--model-path", default=DEFAULT_BASE_MODEL)
    sub_infer.add_argument("--adapter-path", default="")
    sub_infer.add_argument("--text", required=True)
    sub_infer.add_argument("--lang", default="twi")
    sub_infer.add_argument("--ref-audio", required=True)
    sub_infer.add_argument("--ref-text", required=True)
    sub_infer.add_argument("--out-wav", default="output.wav")

    # Command: eval
    sub_eval = subparsers.add_parser("eval", help="Evaluate checkpoint CER with OmniASR 300M")
    sub_eval.add_argument("--checkpoint", required=True)
    sub_eval.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    sub_eval.add_argument("--asr-model", default="/mnt/volume_d2wey28/models/sherpa/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-v2-2026-02-05/model.onnx")
    sub_eval.add_argument("--asr-tokens", default="/mnt/volume_d2wey28/models/sherpa/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-v2-2026-02-05/tokens.txt")
    sub_eval.add_argument("--out-dir", default="eval_samples")

    # Command: run (all-in-one pipeline)
    sub_run = subparsers.add_parser("run", help="Run full pipeline: prep -> encode -> train -> merge")
    sub_run.add_argument("--data-dir", required=True)
    sub_run.add_argument("--lang", required=True)
    sub_run.add_argument("--run-name", required=True)
    sub_run.add_argument("--input-dir", default="")
    sub_run.add_argument("--metadata-file", default="metadata.csv")
    sub_run.add_argument("--hf-repo", default="")
    sub_run.add_argument("--subset", default="")
    sub_run.add_argument("--split", default="train")
    sub_run.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    sub_run.add_argument("--output-dir", default="output")
    sub_run.add_argument("--epochs", type=int, default=20)
    sub_run.add_argument("--learning-rate", type=float, default=5e-5)
    sub_run.add_argument("--batch-size", type=int, default=16, help="Encode batch size")
    sub_run.add_argument("--train-batch-size", type=int, default=8, help="Train batch size")
    sub_run.add_argument("--grad-accum", type=int, default=4)
    sub_run.add_argument("--eval-fraction", type=float, default=0.05)
    sub_run.add_argument("--early-stopping-patience", type=int, default=4)
    sub_run.add_argument("--min-dur", type=float, default=2.0)
    sub_run.add_argument("--max-dur", type=float, default=15.0)
    sub_run.add_argument("--per-lang", type=int, default=0)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "speak":
        from .engine import GhanaVoice
        engine = GhanaVoice(lang=args.lang, model=args.model, device=args.device)
        engine.speak(
            text=args.text,
            output_file=args.out,
            is_ipa=args.ipa,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
    elif args.command == "web":
        from .web import launch_web
        launch_web(port=args.port, share=args.share)
    elif args.command == "serve":
        from .server import start_server
        start_server(host=args.host, port=args.port)
    elif args.command == "export-gguf":
        from .export import export_to_gguf
        export_to_gguf(args.model_dir, args.out_dir, quant_types=args.quant)
    elif args.command == "list-langs":
        print_languages_table(args.tier)
    elif args.command == "prep":
        info = get_language_info(args.lang)
        print(f"\n[Language Info] {info['display']} | Tier {info['tier']} (Target data: {info['rec_hours']})")
        from .prep import prep_local_folder, prep_hf_dataset
        if args.hf_repo:
            prep_hf_dataset(args.hf_repo, args.subset, args.split, args.out_dir, args.lang,
                            args.min_dur, args.max_dur, args.per_lang)
        elif args.input_dir:
            prep_local_folder(args.input_dir, args.metadata_file, args.out_dir, args.lang,
                              args.min_dur, args.max_dur, args.per_lang)
    elif args.command == "encode":
        from .encode import encode_dataset
        encode_dataset(args.data_dir, args.metadata, args.output, args.batch_size)
    elif args.command == "train":
        from .train import train
        train(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            run_name=args.run_name,
            base_model=args.base_model,
            metadata_file=getattr(args, "metadata", "metadata_encoded.csv"),
            no_ref=getattr(args, "no_ref", True),
            full_sft=getattr(args, "full_sft", False),
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            eval_fraction=args.eval_fraction,
            early_stopping_patience=args.early_stopping_patience,
            eval_steps=getattr(args, "eval_steps", 100),
            save_steps=getattr(args, "save_steps", 100),
            logging_steps=getattr(args, "logging_steps", 20),
            eval_cer=getattr(args, "eval_cer", True),
        )
    elif args.command == "merge":
        from .merge import merge_lora
        merge_lora(args.base_model, args.adapter_path, args.output_dir)
    elif args.command == "infer":
        from .infer import synthesize
        synthesize(
            model_path=args.model_path,
            text=args.text,
            ref_audio_path=args.ref_audio,
            ref_text=args.ref_text,
            out_wav_path=args.out_wav,
            lang=args.lang,
            adapter_path=args.adapter_path,
        )
    elif args.command == "eval":
        from .eval import evaluate
        sample = [{
            "lang": "twi",
            "name": "Twi",
            "text": "Onyankopɔn yɛ ɔdɔ na ɔdɔ obiara wɔ wiase.",
            "ref_phones": "n a d a w i d t u m i y e e k e s e e",
            "ref_codes": [100] * 150,
        }]
        evaluate(args.checkpoint, args.base_model, args.asr_model, args.asr_tokens, sample, args.out_dir)
    elif args.command == "run":
        run_pipeline(args)


if __name__ == "__main__":
    main()
