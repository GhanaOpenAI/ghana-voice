"""
Interactive Web Application for Ghana Voice (Gradio).
Supports real-time text-to-speech across Ghanaian languages on CPU or GPU.
"""

import os
import tempfile
import soundfile as sf

from .constants import ALL_SUPPORTED_LANGUAGES, SAMPLE_RATE_24K
from .engine import GhanaVoice

# Cache loaded engines per language & device
_CACHED_ENGINES = {}


def get_engine(lang: str, device: str = "cpu", model: str = None) -> GhanaVoice:
    cache_key = f"{lang}_{device}_{model or 'default'}"
    if cache_key not in _CACHED_ENGINES:
        _CACHED_ENGINES[cache_key] = GhanaVoice(lang=lang, model=model, device=device)
    return _CACHED_ENGINES[cache_key]


def launch_web(port: int = 7860, share: bool = False):
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("Gradio is required for the web UI. Run: pip install gradio")

    SAMPLE_PRESETS = {
        "asante_twi": "Akwaaba, yɛbɔ mpae sɛ Onyankopɔn bɛhyira wo ne wo fiefoɔ nyinaa.",
        "akuapem_twi": "Me kɔ sukuu da biara na mepɛ sɛ mesua ade pa ara.",
        "fante": "Nyame yɛ kɛse na ɔwɔ tumi nyinaa wɔ sor ne fam.",
        "ga": "Nyɔŋmɔ ji mi nɔkwɛlɔ, nɔ ko nɔ ko ehiaŋ mi.",
        "ewe": "Yehowa nye nye kplɔla, naneke mehiãam o.",
        "dagbani": "Naawuni n-nyɛ n gubi, shɛli bi kpaŋsi ma.",
        "ghana_english": "Welcome to Ghana, the gateway to Africa and the home of peace.",
    }

    lang_choices = [(v["display"], k) for k, v in ALL_SUPPORTED_LANGUAGES.items()]

    def generate_speech(text, lang, device, temperature, top_p, repetition_penalty):
        if not text or not text.strip():
            return None, "Please enter some text to synthesize."

        try:
            engine = get_engine(lang=lang, device=device)
            ipa = engine.to_ipa(text)
            wav = engine.synthesize(
                text=text,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, wav, SAMPLE_RATE_24K)
                return f.name, f"Synthesized ({len(wav)/SAMPLE_RATE_24K:.2f}s) | IPA: {ipa}"
        except Exception as e:
            return None, f"Error: {str(e)}"

    def update_sample(lang):
        return SAMPLE_PRESETS.get(lang, "Akwaaba!")

    with gr.Blocks(title="Ghana Voice 🇬🇭") as demo:
        gr.Markdown("# 🇬🇭 Ghana Voice: Text-to-Speech Foundation")
        gr.Markdown(
            "State-of-the-art zero-reference Text-to-Speech for Ghanaian & African languages. "
            "Supports real-time CPU and GPU inference."
        )

        with gr.Row():
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label="Input Text",
                    placeholder="Enter sentence in Ghanaian language or Ghanaian English...",
                    lines=3,
                    value=SAMPLE_PRESETS["asante_twi"],
                )
                with gr.Row():
                    lang_dropdown = gr.Dropdown(
                        choices=lang_choices,
                        value="asante_twi",
                        label="Language",
                    )
                    device_choice = gr.Radio(
                        choices=["cpu", "cuda"],
                        value="cpu",
                        label="Compute Device (CPU / GPU)",
                    )

                with gr.Accordion("Advanced Voice Controls", open=False):
                    temperature = gr.Slider(0.1, 1.2, value=0.7, step=0.05, label="Temperature")
                    top_p = gr.Slider(0.1, 1.0, value=0.8, step=0.05, label="Top-P")
                    rep_pen = gr.Slider(1.0, 1.5, value=1.05, step=0.01, label="Repetition Penalty")

                synth_btn = gr.Button("Synthesize Voice 🔊", variant="primary")

            with gr.Column(scale=2):
                audio_output = gr.Audio(label="Synthesized Speech", type="filepath")
                info_output = gr.Textbox(label="Phonetics & Generation Status", lines=2)

        lang_dropdown.change(fn=update_sample, inputs=lang_dropdown, outputs=text_input)
        synth_btn.click(
            fn=generate_speech,
            inputs=[text_input, lang_dropdown, device_choice, temperature, top_p, rep_pen],
            outputs=[audio_output, info_output],
        )

    demo.launch(server_port=port, share=share)
