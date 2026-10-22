"""
GGUF Model Export and Quantization for ghana-voice models.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

LLAMA_CPP_CONVERT = Path("/mnt/volume_d2wey28/tools/llama.cpp/convert_hf_to_gguf.py")
LLAMA_QUANTIZE = Path("/mnt/volume_d2wey28/tools/llama.cpp/build/bin/llama-quantize")


def export_to_gguf(
    model_dir: str,
    out_dir: str,
    name: Optional[str] = None,
    quant_types: Optional[List[str]] = None,
    clean_f16: bool = True,
) -> List[str]:
    """
    Export a Hugging Face / PyTorch safetensors model to GGUF format and quantize for CPU inference.

    Args:
        model_dir: Directory containing PyTorch safetensors model and tokenizer.
        out_dir: Output directory for GGUF files.
        name: Base filename prefix for GGUF models (e.g. "ghana-voice_twi").
        quant_types: List of quantization types, default: ["Q4_K_M", "Q8_0"].
        clean_f16: Whether to delete the intermediate uncompressed F16 GGUF file.

    Returns:
        List of generated GGUF file paths.
    """
    if quant_types is None:
        quant_types = ["Q4_K_M", "Q8_0"]

    m_path = Path(model_dir)
    o_path = Path(out_dir)
    o_path.mkdir(parents=True, exist_ok=True)

    # Resolve model name: if name given use it, else if directory is 'final', look up parent
    if name:
        base_name = name
    elif m_path.name == "final":
        base_name = m_path.parent.name.replace("neu-tts-ghana-", "ghana-voice_")
    else:
        base_name = m_path.name

    f16_file = o_path / f"{base_name}-f16.gguf"
    generated_files = []

    print("=" * 65)
    print(f"[GGUF Export] Converting {model_dir} to F16 GGUF ...")
    print("=" * 65)

    convert_cmd = [
        "python",
        str(LLAMA_CPP_CONVERT),
        str(m_path),
        "--outtype",
        "f16",
        "--outfile",
        str(f16_file),
    ]

    subprocess.run(convert_cmd, check=True)
    print(f"[GGUF Export] Created F16 base: {f16_file} ({f16_file.stat().st_size / 1e6:.1f} MB)")

    for qtype in quant_types:
        q_file = o_path / f"{base_name}-{qtype}.gguf"
        print(f"[GGUF Export] Quantizing to {qtype} -> {q_file} ...")
        quant_cmd = [str(LLAMA_QUANTIZE), str(f16_file), str(q_file), qtype]
        subprocess.run(quant_cmd, check=True)
        print(f"  -> {qtype} ready ({q_file.stat().st_size / 1e6:.1f} MB)")
        generated_files.append(str(q_file))

    if clean_f16 and f16_file.exists():
        f16_file.unlink()

    print("=" * 65)
    print(f"[GGUF Export] Successfully exported {len(generated_files)} GGUF models!")
    print("=" * 65)
    return generated_files
