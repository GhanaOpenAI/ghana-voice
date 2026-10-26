"""
Standalone voice-conversion worker, run by vc.py under a separate Python
(GHANA_VOICE_VC_PYTHON) whose environment has africa-female-vc / Seed-VC deps.

Protocol: one "<in_wav>\t<out_wav>" per stdin line; replies "ok" or "err <msg>"
on the original stdout. Everything else the libraries print goes to stderr.
Only imports africa-female-vc, never ghana_voice.
"""

import os
import sys

reply = os.fdopen(os.dup(1), "w")
os.dup2(2, 1)


def main():
    voice, steps = sys.argv[1], int(sys.argv[2])
    try:
        import numpy as np
        import soundfile as sf
        from africa_female_vc.engine import VoiceConverter

        vc = VoiceConverter(voice=voice, diffusion_steps=steps, install_deps=False)
        vc.load()
    except Exception as exc:
        reply.write(f"err {exc!r}\n")
        reply.flush()
        return
    reply.write("ready\n")
    reply.flush()

    for line in sys.stdin:
        try:
            src, dst = line.rstrip("\n").split("\t")
            wav, sr = sf.read(src)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            out, out_sr = vc.convert_array(np.asarray(wav, dtype=np.float32), sampling_rate=sr)
            sf.write(dst, out, out_sr)
            reply.write("ok\n")
        except Exception as exc:
            reply.write(f"err {exc!r}\n")
        reply.flush()


if __name__ == "__main__":
    main()
